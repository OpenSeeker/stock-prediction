import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# 导入 Model 用于 Functional API
from tensorflow.keras.models import Sequential, load_model, Model
# 导入 Input 用于 Functional API
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam # 导入 Adam 以便设置学习率
from tensorflow.keras.regularizers import l1_l2 # 导入 L1/L2 正则化器
import ta # 导入技术指标库
import datetime
import warnings
import os
import joblib # 用于保存/加载 scaler
# 导入 scikit-optimize
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import Optimizer # 导入 Optimizer 用于回调中获取结果
import tensorflow as tf
import gc # 导入垃圾回收模块
import traceback # 导入 traceback 用于打印更详细的错误信息

# --- 配置 ---
# 忽略 TensorFlow 的一些警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
# 创建保存模型、scaler、列名和配置的目录
MODEL_SAVE_DIR = "saved_models"
SCALER_SAVE_DIR = "saved_scalers"
COLUMN_SAVE_DIR = "saved_columns"
CONFIG_SAVE_DIR = "saved_configs"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(SCALER_SAVE_DIR, exist_ok=True)
os.makedirs(COLUMN_SAVE_DIR, exist_ok=True)
os.makedirs(CONFIG_SAVE_DIR, exist_ok=True)

# --- 1. 数据获取 ---
def fetch_data(ticker, start_date, end_date):
    """使用 yfinance 获取股票或黄金数据，并处理可能的MultiIndex列"""
    print(f"正在获取 {ticker} 从 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} 的数据...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
             raise ValueError(f"无法获取 {ticker} 的数据，请检查代码是否正确或日期范围内是否有数据。")

        if isinstance(data.columns, pd.MultiIndex):
            print("检测到 MultiIndex 列，正在简化...")
            data.columns = data.columns.get_level_values(0)
            data = data.loc[:,~data.columns.duplicated()]
            print(f"简化后的列名: {list(data.columns)}")

        if 'Adj Close' in data.columns:
            data = data.drop(columns=['Adj Close'])

        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
             raise ValueError(f"获取的数据缺少必要列: {missing_cols}")

        print(f"成功获取 {len(data)} 条数据，使用列: {required_columns}")
        return data[required_columns]
    except Exception as e:
        print(f"获取或处理数据时出错: {e}")
        return None

# --- 2. 数据预处理 (添加指标, 计算百分比变化目标) ---
def preprocess_data_target_pct_change(data):
    """预处理数据：添加技术指标，计算百分比变化作为目标，处理NaN"""
    if data is None:
        print("输入数据为空，无法预处理。")
        return None, None, None

    print("正在添加技术指标 (RSI, MACD)...")
    try:
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        data = data.sort_index()

        if data['Close'].isnull().all() or len(data['Close'].dropna()) < 14:
             print("警告：收盘价数据不足或全为 NaN，无法计算技术指标。")
             return None, None, None

        data['RSI'] = ta.momentum.rsi(data['Close'])
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        data['MACD_diff'] = macd.macd_diff()

        # 添加更多技术指标
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
        bollinger = ta.volatility.BollingerBands(data['Close'])
        data['Bollinger_Upper'] = bollinger.bollinger_hband()
        data['Bollinger_Lower'] = bollinger.bollinger_lband()
        data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()

        data['Target_Pct_Change'] = data['Close'].pct_change().shift(-1)

        initial_len = len(data)
        data.dropna(inplace=True)
        final_len = len(data)
        print(f"添加指标/目标并移除 NaN 后，数据从 {initial_len} 条变为 {final_len} 条。")

        if final_len == 0:
            print("处理后数据为空。")
            return None, None, None

    except Exception as e:
        print(f"计算或添加技术指标/目标时出错: {e}")
        traceback.print_exc()
        return None, None, None

    target_series = data['Target_Pct_Change']
    feature_df = data.drop(columns=['Target_Pct_Change'])
    feature_columns = list(feature_df.columns)
    num_features = len(feature_columns)
    print(f"使用的特征列 ({num_features} 个): {feature_columns}")

    return feature_df, target_series, feature_columns

# --- 3. 创建时间序列数据 (特征和目标分开处理) ---
def create_sequences_feature_target(scaled_features, scaled_target, look_back):
    """从缩放后的特征和目标数据创建时间序列 X 和 y"""
    X, y = [], []
    if len(scaled_features) != len(scaled_target):
        print("错误：缩放后的特征和目标长度不一致！")
        return None, None
    if len(scaled_features) < look_back:
        print(f"数据长度 ({len(scaled_features)}) 小于 look_back ({look_back})，无法创建序列。")
        return None, None

    print(f"正在从 {len(scaled_features)} 条数据创建 look_back={look_back} 的时间序列...")
    for i in range(look_back, len(scaled_features)):
        X.append(scaled_features[i-look_back:i, :])
        y.append(scaled_target[i])

    if not X or not y:
        print("创建时间序列失败。")
        return None, None
    print("时间序列创建完成。")
    return np.array(X), np.array(y)

# --- 4. 模型构建 (接受超参数, 添加正则化) ---
def build_model_for_tuning(look_back, num_features, units, dropout_rate, learning_rate,
                           l1_reg, l2_reg, recurrent_dropout_rate):
    """构建用于超参数优化的模型 (包含正则化)"""
    units_int = int(units)
    regularizer = l1_l2(l1=l1_reg, l2=l2_reg)

    inputs = Input(shape=(look_back, num_features))
    x = LSTM(units=units_int, return_sequences=True,
             kernel_regularizer=regularizer,
             recurrent_dropout=recurrent_dropout_rate)(inputs)
    x = Dropout(dropout_rate)(x)
    x = LSTM(units=units_int, return_sequences=False,
             kernel_regularizer=regularizer,
             recurrent_dropout=recurrent_dropout_rate)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(units=50, activation='relu',
              kernel_regularizer=regularizer)(x)
    outputs = Dense(units=1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# --- 5. 目标函数 (使用单次验证集, 目标为百分比变化) ---
data_cache = {}

def objective_function(ticker, train_years, look_back, units, dropout_rate, learning_rate, batch_size,
                       l1_reg, l2_reg, recurrent_dropout_rate):
    """用于贝叶斯优化的目标函数，使用单次验证集，返回负的验证方向准确率"""
    print(f"\n--- 尝试配置: train_years={train_years}, look_back={look_back}, units={units}, dropout={dropout_rate:.4f}, recurrent_dropout={recurrent_dropout_rate:.4f}, lr={learning_rate:.6f}, batch_size={batch_size}, l1={l1_reg:.6f}, l2={l2_reg:.6f} ---")

    data_key = (ticker, train_years)
    if data_key in data_cache:
        print("从缓存加载数据...")
        data_unprocessed = data_cache[data_key].copy()
    else:
        train_years_int = int(train_years)
        end_date = datetime.datetime.now() - datetime.timedelta(days=1)
        start_date = end_date - datetime.timedelta(days=365 * train_years_int)
        data_unprocessed = fetch_data(ticker, start_date, end_date)
        if data_unprocessed is None:
            print("获取数据失败，跳过此配置。")
            return 1.0
        data_cache[data_key] = data_unprocessed.copy()

    feature_df, target_series, feature_columns = preprocess_data_target_pct_change(data_unprocessed)
    if feature_df is None or target_series is None or len(feature_df) < look_back + 1:
        print("预处理失败或数据不足，跳过此配置。")
        return 1.0

    num_features = len(feature_columns)

    print("正在缩放特征和目标数据...")
    feature_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(feature_df)
    target_scaler = StandardScaler()
    scaled_target = target_scaler.fit_transform(target_series.values.reshape(-1, 1)).flatten()

    X, y = create_sequences_feature_target(scaled_features, scaled_target, look_back)
    if X is None:
        print("创建序列失败，跳过此配置。")
        return 1.0

    if len(X) < 2:
        print("序列数据过少，无法分割，跳过此配置。")
        return 1.0
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    if len(X_train) == 0 or len(X_val) == 0:
        print("分割后训练集或验证集为空，跳过此配置。")
        return 1.0

    model = build_model_for_tuning(look_back, num_features, units, dropout_rate, learning_rate,
                                   l1_reg, l2_reg, recurrent_dropout_rate)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    print("开始训练 (优化运行)...")
    history = model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping],
                        verbose=1)

    min_val_loss = min(history.history['val_loss']) if 'val_loss' in history.history and history.history['val_loss'] else 100.0
    print(f"配置完成。最小验证损失 (预测缩放后的百分比变化): {min_val_loss:.6f}")

    print("正在评估验证集方向准确率...")
    directional_accuracy_val = 0.0
    try:
        predictions_scaled_val = model.predict(X_val, verbose=0)
        predictions_pct_change_val = target_scaler.inverse_transform(predictions_scaled_val).flatten()

        val_start_index = len(X_train)
        actual_pct_change_val = target_series.iloc[val_start_index:].values[:len(X_val)]

        if len(predictions_pct_change_val) != len(actual_pct_change_val):
             print(f"警告：预测百分比变化长度 ({len(predictions_pct_change_val)}) 与实际百分比变化长度 ({len(actual_pct_change_val)}) 不匹配！跳过准确率计算。")
        else:
            correct_direction_count_val = 0
            for i in range(len(predictions_pct_change_val)):
                pred_sign = np.sign(predictions_pct_change_val[i])
                actual_sign = np.sign(actual_pct_change_val[i])
                if pred_sign == actual_sign:
                    correct_direction_count_val += 1
            directional_accuracy_val = correct_direction_count_val / len(predictions_pct_change_val) if len(predictions_pct_change_val) > 0 else 0.0
            print(f"本次配置验证集方向准确率: {directional_accuracy_val * 100:.2f}%")

    except Exception as e:
        print(f"计算验证集准确率时出错: {e}")
        traceback.print_exc()

    tf.keras.backend.clear_session()
    # 移除 del 语句以避免 UnboundLocalError
    gc.collect()

    return -directional_accuracy_val

# --- 6. 运行贝叶斯优化 (自动保存历史最佳配置) ---
def run_bayesian_optimization(ticker, n_calls=20):
    """运行贝叶斯优化来寻找最佳超参数 (不使用CV, 含正则化, 目标: 百分比变化准确率, 自动保存历史最佳)"""
    search_space = [
        Integer(2, 8, name='train_years'),
        Integer(30, 120, name='look_back'),
        Integer(50, 250, name='units'),
        Real(0.1, 0.5, prior='uniform', name='dropout_rate'),
        Real(0.0, 0.5, prior='uniform', name='recurrent_dropout_rate'),
        Real(1e-4, 1e-2, prior='log-uniform', name='learning_rate'),
        Categorical([16, 32, 64], name='batch_size'),
        Real(1e-6, 1e-1, prior='log-uniform', name='l1_reg'),
        Real(1e-6, 1e-1, prior='log-uniform', name='l2_reg')
    ]

    # 定义保存最佳配置的文件路径
    best_config_path = os.path.join(CONFIG_SAVE_DIR, f"{ticker}_best_config_tech_reg_pct.joblib")
    previous_best_accuracy = -1.0 # 初始化为最低可能准确率的负值
    previous_best_params = None

    # 尝试加载之前的最佳配置
    if os.path.exists(best_config_path):
        try:
            print(f"加载之前的最佳配置: {best_config_path}")
            # 注意：旧的配置文件可能没有 'best_accuracy' 字段
            loaded_config = joblib.load(best_config_path)
            # 假设旧配置文件只存了参数，我们需要一个方法来获取其对应的准确率
            # 为了简化，我们这里假设旧文件包含 'best_accuracy'，如果不存在则忽略
            if isinstance(loaded_config, dict) and 'best_accuracy' in loaded_config:
                 previous_best_accuracy = loaded_config['best_accuracy']
                 previous_best_params = {k: v for k, v in loaded_config.items() if k != 'best_accuracy'}
                 print(f"之前最佳准确率: {previous_best_accuracy * 100:.2f}%")
            elif isinstance(loaded_config, dict): # 如果只存了参数
                 print("警告：旧配置文件缺少 'best_accuracy' 字段，无法比较，将覆盖。")
                 previous_best_params = loaded_config # 仍然加载参数以备后用
                 previous_best_accuracy = -1.0 # 强制覆盖
            else:
                 print("警告：无法识别旧配置文件格式，将覆盖。")
        except Exception as e:
            print(f"加载旧配置文件时出错: {e}，将覆盖。")
            previous_best_accuracy = -1.0
            previous_best_params = None


    checkpoint = {'best_accuracy': previous_best_accuracy, 'best_params': previous_best_params} # 初始化 checkpoint

    def optimization_callback(res):
        nonlocal previous_best_accuracy, previous_best_params # 允许修改外部变量
        checkpoint['n_calls'] = len(res.x_iters)
        current_best_neg_accuracy = res.fun
        current_best_accuracy = -current_best_neg_accuracy
        current_best_params_list = res.x
        current_best_params_dict = {dim.name: val for dim, val in zip(search_space, current_best_params_list)}

        print(f"\n[优化回调 - 第 {checkpoint['n_calls']}/{n_calls} 次尝试后]")
        print(f"  当前最佳验证准确率: {current_best_accuracy * 100:.2f}% (目标函数值: {current_best_neg_accuracy:.6f})")
        print(f"  对应的最佳参数: {current_best_params_dict}")

        # 更新 checkpoint 中的最佳记录 (无论是否优于历史最佳)
        checkpoint['best_params'] = current_best_params_dict
        checkpoint['best_accuracy'] = current_best_accuracy

    @use_named_args(search_space)
    def wrapped_objective(**params):
        return objective_function(ticker=ticker, **params)

    print(f"\n--- 开始对 {ticker} 进行贝叶斯超参数优化 (尝试 {n_calls} 次, 含正则化, 目标: 最大化验证准确率(基于百分比变化)) ---")
    global data_cache
    data_cache = {}

    result = gp_minimize(
        func=wrapped_objective,
        dimensions=search_space,
        n_calls=n_calls,
        callback=[optimization_callback],
        random_state=42,
        n_initial_points=5
    )

    print("\n--- 贝叶斯优化完成 ---")
    # 从 checkpoint 获取本次运行的最佳结果
    new_best_params_dict = checkpoint.get('best_params', None)
    new_best_accuracy = checkpoint.get('best_accuracy', -1.0)

    if new_best_params_dict:
        print(f"本次优化找到的最佳参数组合:")
        print(new_best_params_dict)
        print(f"对应的验证准确率: {new_best_accuracy * 100:.2f}%")

        # 比较并决定是否保存
        if new_best_accuracy > previous_best_accuracy:
            print(f"新结果 ({new_best_accuracy*100:.2f}%) 优于之前的最佳结果 ({previous_best_accuracy*100:.2f}% 或未记录)。正在保存新配置...")
            try:
                # 保存时不包含准确率，只保存参数
                params_to_save = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) for k, v in new_best_params_dict.items()}
                joblib.dump(params_to_save, best_config_path)
                print(f"最佳配置已更新并保存至: {best_config_path}")
            except Exception as e:
                print(f"保存最佳配置时出错: {e}")
            return new_best_params_dict # 返回新找到的最佳参数，无论是否保存成功
        else:
            print(f"新结果 ({new_best_accuracy*100:.2f}%) 不优于之前的最佳结果 ({previous_best_accuracy*100:.2f}%). 保留之前的配置。")
            if previous_best_params:
                 print("将使用之前保存的最佳参数进行后续操作。")
                 return previous_best_params
            else:
                 print("警告：没有找到之前的最佳参数，将使用本次运行找到的参数。")
                 return new_best_params_dict # 即使没有历史最佳，也返回本次找到的
    else:
        print("警告：优化过程未能找到有效的最佳参数。")
        try:
             best_params_dict_from_result = {dim.name: val for dim, val in zip(search_space, result.x)}
             print(f"从 gp_minimize 最终结果获取的参数: {best_params_dict_from_result}")
             return best_params_dict_from_result
        except:
             print("无法从优化结果中获取参数。")
             return None

# --- 7. 使用最佳参数训练最终模型 (目标: 百分比变化) ---
def train_final_model(ticker, best_params):
    """使用找到的最佳参数训练最终模型并保存 (目标: 百分比变化)"""
    # ... (函数内部逻辑不变，但移除保存 config 的代码) ...
    train_years = best_params['train_years']
    look_back = best_params['look_back']
    units = best_params['units']
    dropout_rate = best_params['dropout_rate']
    learning_rate = best_params['learning_rate']
    batch_size = best_params['batch_size']
    l1_reg = best_params['l1_reg']
    l2_reg = best_params['l2_reg']
    recurrent_dropout_rate = best_params['recurrent_dropout_rate']

    print(f"\n--- 使用最佳参数为 {ticker} 训练最终模型 (目标: 百分比变化) ---")
    print(f"参数: train_years={train_years}, look_back={look_back}, units={units}, dropout={dropout_rate:.4f}, recurrent_dropout={recurrent_dropout_rate:.4f}, lr={learning_rate:.6f}, batch_size={batch_size}, l1={l1_reg:.6f}, l2={l2_reg:.6f}")

    train_years_int = int(train_years)
    end_date = datetime.datetime.now() - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=365 * train_years_int)

    data_unprocessed = fetch_data(ticker, start_date, end_date)
    if data_unprocessed is None: return 0.0

    feature_df, target_series, feature_columns = preprocess_data_target_pct_change(data_unprocessed.copy())
    if feature_df is None or target_series is None or len(feature_df) < look_back + 1:
        print("最终训练数据预处理失败或数据不足。")
        return 0.0

    print("最终训练：缩放特征和目标数据...")
    feature_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(feature_df)
    target_scaler = StandardScaler()
    scaled_target = target_scaler.fit_transform(target_series.values.reshape(-1, 1)).flatten()

    X, y = create_sequences_feature_target(scaled_features, scaled_target, look_back)
    if X is None: return 0.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    num_features = X_train.shape[2]

    if len(X_train) == 0 or len(X_test) == 0:
        print("分割后训练集或测试集为空，无法训练最终模型。")
        return 0.0

    model = build_model_for_tuning(look_back, num_features, units, dropout_rate, learning_rate,
                                   l1_reg, l2_reg, recurrent_dropout_rate)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-6, verbose=1)

    print("开始训练最终模型...")
    history = model.fit(X_train, y_train,
                        epochs=150,
                        batch_size=batch_size,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping, reduce_lr],
                        verbose=1)

    print("最终模型训练完成。")

    print("正在评估最终模型在测试集上的方向准确率...")
    directional_accuracy = 0.0
    try:
        predictions_scaled = model.predict(X_test, verbose=0)
        predictions_pct_change = target_scaler.inverse_transform(predictions_scaled).flatten()

        test_start_index = len(X_train)
        actual_pct_change_test = target_series.iloc[test_start_index:].values[:len(X_test)]

        if len(predictions_pct_change) != len(actual_pct_change_test):
             print(f"警告：测试集预测百分比变化长度 ({len(predictions_pct_change)}) 与实际百分比变化长度 ({len(actual_pct_change_test)}) 不匹配！")
        else:
            correct_direction_count = 0
            for i in range(len(predictions_pct_change)):
                pred_sign = np.sign(predictions_pct_change[i])
                actual_sign = np.sign(actual_pct_change_test[i])
                if pred_sign == actual_sign:
                    correct_direction_count += 1
            directional_accuracy = correct_direction_count / len(predictions_pct_change) if len(predictions_pct_change) > 0 else 0.0
            print(f"最终模型测试集方向准确率: {directional_accuracy * 100:.2f}%")
    except Exception as e:
        print(f"评估最终模型准确率时出错: {e}")
        traceback.print_exc()

    # --- 保存最终模型、Scaler、列名 (不再保存配置) ---
    model_path = os.path.join(MODEL_SAVE_DIR, f"{ticker}_best_model_tech_reg_pct.keras")
    scaler_features_path = os.path.join(SCALER_SAVE_DIR, f"{ticker}_best_scaler_features_tech_reg_pct.joblib")
    scaler_target_path = os.path.join(SCALER_SAVE_DIR, f"{ticker}_best_scaler_target_tech_reg_pct.joblib")
    columns_path = os.path.join(COLUMN_SAVE_DIR, f"{ticker}_best_columns_tech_reg_pct.joblib")
    # config_path = os.path.join(CONFIG_SAVE_DIR, f"{ticker}_best_config_tech_reg_pct.joblib") # 移除配置保存
    try:
        model.save(model_path)
        joblib.dump(feature_scaler, scaler_features_path)
        joblib.dump(target_scaler, scaler_target_path)
        joblib.dump(feature_columns, columns_path)
        # best_params_to_save = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) for k, v in best_params.items()}
        # joblib.dump(best_params_to_save, config_path) # 移除配置保存
        print(f"最终模型已保存至: {model_path}")
        print(f"特征 Scaler 已保存至: {scaler_features_path}")
        print(f"目标 Scaler 已保存至: {scaler_target_path}")
        print(f"列名已保存至: {columns_path}")
        # print(f"最佳配置已保存至: {config_path}") # 移除
    except Exception as e:
        print(f"保存最终模型、Scaler或列名时出错: {e}") # 更新错误信息
        return 0.0

    return directional_accuracy

# --- 8. 加载最终模型、Scaler和配置 (目标: 百分比变化) ---
def load_best_model_scaler_config(ticker):
    """加载指定ticker的最终优化模型、scaler和配置 (目标: 百分比变化)"""
    model_path = os.path.join(MODEL_SAVE_DIR, f"{ticker}_best_model_tech_reg_pct.keras")
    scaler_features_path = os.path.join(SCALER_SAVE_DIR, f"{ticker}_best_scaler_features_tech_reg_pct.joblib")
    scaler_target_path = os.path.join(SCALER_SAVE_DIR, f"{ticker}_best_scaler_target_tech_reg_pct.joblib")
    config_path = os.path.join(CONFIG_SAVE_DIR, f"{ticker}_best_config_tech_reg_pct.joblib")
    columns_path = os.path.join(COLUMN_SAVE_DIR, f"{ticker}_best_columns_tech_reg_pct.joblib")

    required_files = [model_path, scaler_features_path, scaler_target_path, config_path, columns_path]
    if not all(os.path.exists(p) for p in required_files):
        print(f"未找到 {ticker} 的已保存最终优化模型(目标pct)、scaler、配置或列名文件。请先运行优化和训练。")
        return None, None, None, None, None

    try:
        print(f"正在加载最终模型(目标pct): {model_path}")
        model = load_model(model_path)
        print(f"正在加载特征 scaler: {scaler_features_path}")
        feature_scaler = joblib.load(scaler_features_path)
        print(f"正在加载目标 scaler: {scaler_target_path}")
        target_scaler = joblib.load(scaler_target_path)
        print(f"正在加载配置: {config_path}")
        config = joblib.load(config_path) # 加载的配置现在只包含参数
        print(f"正在加载列名: {columns_path}")
        columns = joblib.load(columns_path)

        look_back = config.get('look_back', 60)
        print(f"加载成功。模型配置: look_back={look_back}, features={columns}")
        print(f"完整配置: {config}")
        return model, feature_scaler, target_scaler, config, columns
    except Exception as e:
        print(f"加载最终模型、scaler、配置或列名时出错: {e}")
        traceback.print_exc()
        return None, None, None, None, None

# --- 9. 预测 (使用加载的最终模型，目标: 百分比变化) ---
# predict_pct_change_final, predict_next_trading_day_final, predict_specific_date_final 函数保持不变

def predict_pct_change_final(model, feature_scaler, target_scaler, input_data_sequence, columns):
    """使用加载的最终模型预测百分比变化"""
    if model is None or feature_scaler is None or target_scaler is None or input_data_sequence is None: 
        return None
    try:
        prediction_scaled = model.predict(input_data_sequence, verbose=0)
        prediction_pct_change = target_scaler.inverse_transform(prediction_scaled).flatten()[0]
        return prediction_pct_change
    except Exception as e: 
        print(f"预测百分比变化时发生错误: {e}")
        return None

def predict_price_final(model, scaler, input_data_sequence, columns):
    """使用加载的最终模型进行预测"""
    if model is None or scaler is None or input_data_sequence is None or columns is None: return None
    try:
        num_features = len(columns)
        close_col_index = columns.index('Close')
        prediction_scaled = model.predict(input_data_sequence, verbose=0)
        dummy_prediction = np.zeros((1, num_features))
        dummy_prediction[0, close_col_index] = prediction_scaled.flatten()[0]
        prediction = scaler.inverse_transform(dummy_prediction)[0, close_col_index]
        return prediction
    except Exception as e: print(f"预测时发生错误: {e}"); return None

def predict_next_trading_day_final(ticker):
    """使用最终优化模型预测下一交易日 (目标: 百分比变化)"""
    model, feature_scaler, target_scaler, config, columns = load_best_model_scaler_config(ticker)
    if model is None: return

    look_back = config['look_back']
    num_features = len(columns)

    required_history_days = look_back + 60
    end_date = datetime.datetime.now() - datetime.timedelta(days=1) # 确保 end_date 是 вчера
    start_date = end_date - datetime.timedelta(days=required_history_days + 60) # 增加追溯天数

    print(f"为预测获取最新数据 ({ticker})...")
    latest_data_raw = fetch_data(ticker, start_date, end_date)
    if latest_data_raw is None: print("无法获取最新数据。"); return

    print("正在为最新数据添加技术指标...")
    try:
        latest_data_processed, _, _ = preprocess_data_target_pct_change(latest_data_raw.copy())
        if latest_data_processed is None: return
    except Exception as e:
        print(f"为最新数据计算指标时出错: {e}"); return

    if len(latest_data_processed) < look_back:
        print(f"处理后最新数据不足 {look_back} 天 ({len(latest_data_processed)})，无法预测。"); return

    try:
        latest_features = latest_data_processed[columns].iloc[-look_back:]
        last_actual_close_price_before_target = latest_features['Close'].iloc[-1]
        last_actual_date_before_target = latest_features.index[-1]
    except KeyError as e:
        print(f"错误：加载的列名 {columns} 与获取的最新数据列不匹配: {e}"); return
    except IndexError:
         print("错误：无法获取最后一个收盘价。")
         return

    try:
        last_sequence_scaled = feature_scaler.transform(latest_features.values)
    except ValueError as e:
         print(f"使用加载的特征 scaler 缩放最新数据时出错: {e}. 可能特征数量不匹配。"); return
    last_sequence_reshaped = np.reshape(last_sequence_scaled, (1, look_back, num_features))

    next_trading_day = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    predicted_pct_change = predict_pct_change_final(model, feature_scaler, target_scaler, last_sequence_reshaped, columns)

    if predicted_pct_change is not None:
        print(f"\n--- {ticker} 在 {next_trading_day} 的预测 (目标: 百分比变化) ---")
        print(f"预测百分比变化: {predicted_pct_change * 100:.4f}%")
        predicted_price = last_actual_close_price_before_target * (1 + predicted_pct_change)
        print(f"预测收盘价: {predicted_price:.2f}")
        # 打印正确的基准日期和价格
        print(f"目标日期前一交易日 ({last_actual_date_before_target.strftime('%Y-%m-%d')}) 收盘价: {last_actual_close_price_before_target:.2f}")

        if predicted_pct_change > 0: print("预测趋势: 上涨 ▲")
        elif predicted_pct_change < 0: print("预测趋势: 下跌 ▼")
        else: print("预测趋势: 持平 –")
    else: print("无法完成预测。")

def predict_date_range_final(ticker, start_date_str, end_date_str):
    """使用最终优化模型递归预测指定日期范围 (目标: 百分比变化)"""
    model, feature_scaler, target_scaler, config, columns = load_best_model_scaler_config(ticker)
    if model is None: return None

    look_back = config['look_back']
    num_features = len(columns)

    try:
        start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError:
        print("日期格式错误，请输入 YYYY-MM-DD 格式。"); return None

    if start_date > end_date:
        print("开始日期不能晚于结束日期。"); return None

    # 1. 获取足够的初始历史数据 (确保行数足够)
    min_raw_rows_needed = look_back + 40 # 需要 look_back 行用于输入，+40 行缓冲用于指标计算和NaN移除
    initial_end_date = start_date - datetime.timedelta(days=1)
    initial_data_raw = None
    fetch_attempts = 0
    max_fetch_attempts = 5 # 防止无限循环
    days_to_fetch = look_back + 160 # 初始尝试获取的天数 (look_back + 100 buffer + 60 extra)

    print(f"正在获取 {initial_end_date.strftime('%Y-%m-%d')} 及之前的初始历史数据 (目标: 至少 {min_raw_rows_needed} 行)...")

    while fetch_attempts < max_fetch_attempts:
        fetch_attempts += 1
        current_start_date = initial_end_date - datetime.timedelta(days=days_to_fetch)
        print(f"  尝试获取 {days_to_fetch} 天的数据 (从 {current_start_date.strftime('%Y-%m-%d')} 到 {initial_end_date.strftime('%Y-%m-%d')})...")
        
        fetched_data = fetch_data(ticker, current_start_date, initial_end_date)

        if fetched_data is not None and len(fetched_data) >= min_raw_rows_needed:
            initial_data_raw = fetched_data
            print(f"  成功获取 {len(initial_data_raw)} 行原始数据。")
            break
        elif fetched_data is not None:
             print(f"  获取到 {len(fetched_data)} 行，少于所需的 {min_raw_rows_needed} 行。正在尝试获取更早的数据...")
             days_to_fetch += 90 # 增加获取天数，尝试获取更早的数据
        else:
             print(f"  第 {fetch_attempts} 次获取数据失败。")
             # 如果多次失败，可能无法获取数据
             if fetch_attempts >= max_fetch_attempts:
                 print("多次尝试后仍无法获取足够的初始历史数据。")
                 return None
             days_to_fetch += 90 # 即使失败也增加天数，以防是短期问题

    if initial_data_raw is None:
         print("最终未能获取足够的初始历史数据。")
         return None

    print("正在为初始历史数据添加技术指标...")
    try:
        # 注意：这里不计算 Target_Pct_Change，因为我们只需要特征
        initial_data_with_indicators = initial_data_raw.copy()
        if not isinstance(initial_data_with_indicators.index, pd.DatetimeIndex):
            initial_data_with_indicators.index = pd.to_datetime(initial_data_with_indicators.index)
        initial_data_with_indicators = initial_data_with_indicators.sort_index()

        if initial_data_with_indicators['Close'].isnull().all() or len(initial_data_with_indicators['Close'].dropna()) < 35: # MACD 需要更多数据
             print("警告：初始数据收盘价不足或全为 NaN，无法计算技术指标。")
             return None

        initial_data_with_indicators['RSI'] = ta.momentum.rsi(initial_data_with_indicators['Close'])
        macd = ta.trend.MACD(initial_data_with_indicators['Close'])
        initial_data_with_indicators['MACD'] = macd.macd()
        initial_data_with_indicators['MACD_signal'] = macd.macd_signal()
        initial_data_with_indicators['MACD_diff'] = macd.macd_diff()

        initial_len = len(initial_data_with_indicators)
        initial_data_with_indicators.dropna(inplace=True) # 移除计算指标产生的 NaN
        final_len = len(initial_data_with_indicators)
        print(f"为初始数据添加指标并移除 NaN 后，数据从 {initial_len} 条变为 {final_len} 条。")

        if final_len < look_back:
            print(f"处理后的初始历史数据不足 {look_back} 天 ({final_len})，无法开始预测。"); return None

        # 确保列顺序与加载的列名一致
        initial_data_processed = initial_data_with_indicators[columns]

    except Exception as e:
        print(f"为初始历史数据计算指标时出错: {e}");
        traceback.print_exc()
        return None

    # 2. 递归预测循环
    predictions = {}
    current_data_history = initial_data_processed.copy() # 使用处理后的初始数据开始
    current_date = start_date

    while current_date <= end_date:
        target_date_str = current_date.strftime('%Y-%m-%d')
        print(f"\n--- 预测 {ticker} 在 {target_date_str} ---")

        # 检查当天是否为周末 (可选，取决于是否需要跳过)
        # if current_date.weekday() >= 5: # 5 = Saturday, 6 = Sunday
        #     print(f"{target_date_str} 是周末，跳过预测。")
        #     current_date += datetime.timedelta(days=1)
        #     continue

        # a. 准备输入序列
        if len(current_data_history) < look_back:
             print(f"错误：内部错误，历史数据长度 ({len(current_data_history)}) 小于 look_back ({look_back})，无法预测 {target_date_str}。")
             # 尝试继续下一天，但可能指示逻辑错误
             current_date += datetime.timedelta(days=1)
             continue

        input_features = current_data_history[columns].iloc[-look_back:]
        last_known_close = input_features['Close'].iloc[-1]
        last_known_date = input_features.index[-1]

        # b. 缩放输入序列
        try:
            sequence_scaled = feature_scaler.transform(input_features.values)
            sequence_reshaped = np.reshape(sequence_scaled, (1, look_back, num_features))
        except Exception as e:
            print(f"缩放输入序列时出错 ({target_date_str}): {e}");
            current_date += datetime.timedelta(days=1)
            continue

        # c. 进行预测 (百分比变化)
        predicted_pct_change = predict_pct_change_final(model, feature_scaler, target_scaler, sequence_reshaped, columns)

        if predicted_pct_change is None:
            print(f"无法完成对 {target_date_str} 的预测。")
            predictions[target_date_str] = None
            current_date += datetime.timedelta(days=1)
            continue

        # d. 计算预测价格
        predicted_price = last_known_close * (1 + predicted_pct_change)

        print(f"预测百分比变化: {predicted_pct_change * 100:.4f}%")
        print(f"预测收盘价: {predicted_price:.2f}")
        print(f"基于前一可用日期 ({last_known_date.strftime('%Y-%m-%d')}) 的收盘价: {last_known_close:.2f}")
        if predicted_pct_change > 0: print("预测趋势: 上涨 ▲")
        elif predicted_pct_change < 0: print("预测趋势: 下跌 ▼")
        else: print("预测趋势: 持平 –")

        predictions[target_date_str] = {
            'predicted_pct_change': predicted_pct_change,
            'predicted_price': predicted_price,
            'last_actual_close_price_before_target': last_known_close, # 使用 last_known_close
            'last_actual_date_before_target': last_known_date.strftime('%Y-%m-%d') # 使用 last_known_date
        }

        # e. 合成下一天的数据行 (用于下一次预测的输入)
        synthesized_row = pd.DataFrame(index=[current_date])
        # --- Refined OHLC + Trend Extrapolation Synthesis ---
        # Refined OHLC Synthesis
        synthesized_row['Open'] = last_known_close # Assume Open is the last known close
        synthesized_row['High'] = max(last_known_close, predicted_price) * 1.001 # High based on range + margin
        synthesized_row['Low'] = min(last_known_close, predicted_price) * 0.999 # Low based on range - margin
        synthesized_row['Close'] = predicted_price

        # Keep Trend Extrapolation for Volume and Indicators
        if len(input_features) >= 2:
            last_day = input_features.iloc[-1]
            prev_day = input_features.iloc[-2]
            volume_diff = last_day['Volume'] - prev_day['Volume']
            synthesized_row['Volume'] = last_day['Volume'] + volume_diff

            if 'RSI' in columns:
                rsi_diff = last_day['RSI'] - prev_day['RSI']
                synthesized_row['RSI'] = last_day['RSI'] + rsi_diff
            if 'MACD' in columns:
                macd_diff_val = last_day['MACD'] - prev_day['MACD'] # Name conflict with MACD_diff column
                synthesized_row['MACD'] = last_day['MACD'] + macd_diff_val
            if 'MACD_signal' in columns:
                signal_diff = last_day['MACD_signal'] - prev_day['MACD_signal']
                synthesized_row['MACD_signal'] = last_day['MACD_signal'] + signal_diff
            if 'MACD_diff' in columns: # The actual MACD difference column
                diff_col_diff = last_day['MACD_diff'] - prev_day['MACD_diff']
                synthesized_row['MACD_diff'] = last_day['MACD_diff'] + diff_col_diff
        else: # Fallback if only 1 day in history (shouldn't happen with look_back)
            synthesized_row['Volume'] = input_features['Volume'].iloc[-1]
            if 'RSI' in columns: synthesized_row['RSI'] = input_features['RSI'].iloc[-1]
            if 'MACD' in columns: synthesized_row['MACD'] = input_features['MACD'].iloc[-1]
            if 'MACD_signal' in columns: synthesized_row['MACD_signal'] = input_features['MACD_signal'].iloc[-1]
            if 'MACD_diff' in columns: synthesized_row['MACD_diff'] = input_features['MACD_diff'].iloc[-1]
        # --- End Trend Extrapolation Synthesis ---

        # 确保合成行的列顺序与 columns 一致
        synthesized_row = synthesized_row[columns]

        # f. 将合成行添加到历史数据中
        # 使用 concat 而不是 append
        current_data_history = pd.concat([current_data_history, synthesized_row])

        # g. 移动到下一天
        current_date += datetime.timedelta(days=1)

    return predictions


# --- 10. 菜单 ---
def display_menu():
    """显示主菜单"""
    print("\n--- 股票/黄金趋势预测系统 ---")
    print("1. 训练最终模型 (使用上次优化配置)")
    print("2. 预测下一交易日 (使用上次优化模型)")
    print("3. 预测指定日期范围 （误差极大）")
    print("4. 运行贝叶斯超参数优化 (目标: 百分比变化准确率)")
    print("5. 退出")
    print("-----------------------------")

def main():
    """主程序循环"""
    while True:
        display_menu()
        choice = input("请输入选项 (1-5): ")

        if choice == '1': # 训练模型 (使用最佳配置)
            ticker = input("请输入要训练的股票/黄金代码 (例如 'AAPL', 'GC=F'): ").strip().upper()
            if not ticker: print("代码不能为空。"); continue

            best_config_path = os.path.join(CONFIG_SAVE_DIR, f"{ticker}_best_config_tech_reg_pct.joblib")
            if os.path.exists(best_config_path):
                try:
                    best_params = joblib.load(best_config_path)
                    print(f"找到已保存的最佳配置(目标pct)，将使用此配置训练最终模型: {best_params}")
                    train_final_model(ticker, best_params)
                except Exception as e:
                    print(f"加载最佳配置时出错: {e}。")
                    print("请先运行选项 4 进行超参数优化。")
            else:
                print(f"未找到 {ticker} 的最佳配置文件(目标pct)。请先运行选项 4 进行超参数优化。")

        elif choice == '2': # 预测下一交易日 (使用最佳模型)
            ticker = input("请输入要预测的股票/黄金代码 (例如 'AAPL', 'GC=F'): ").strip().upper()
            if not ticker: print("代码不能为空。"); continue
            try: predict_next_trading_day_final(ticker)
            except Exception as e: print(f"预测下一交易日时发生错误: {e}"); import traceback; traceback.print_exc()

        elif choice == '3': # 预测指定日期范围 (使用最佳模型)
            ticker = input("请输入要预测的股票/黄金代码 (例如 'AAPL', 'GC=F'): ").strip().upper()
            if not ticker: print("代码不能为空。"); continue
            start_date_str = input("请输入预测开始日期 (YYYY-MM-DD): ").strip()
            end_date_str = input("请输入预测结束日期 (YYYY-MM-DD): ").strip()
            try:
                # 调用预测函数，详细打印已在函数内部完成
                predictions = predict_date_range_final(ticker, start_date_str, end_date_str)
                if predictions is None: # 检查函数是否因错误返回 None
                    print("日期范围预测未能完成或遇到错误。")
                elif not predictions: # 检查字典是否为空 (例如，如果所有天都预测失败)
                     print("未能生成任何预测结果。")
                # 移除 main 函数中的重复打印循环
            except Exception as e:
                print(f"预测日期范围时发生错误: {e}"); import traceback; traceback.print_exc()

        elif choice == '4': # 运行贝叶斯优化
            ticker = input("请输入要进行优化的股票/黄金代码 (例如 'AAPL', 'GC=F'): ").strip().upper()
            if not ticker: print("代码不能为空。"); continue
            while True:
                try:
                    n_calls_input = input("请输入优化尝试次数 (例如 20 或 50): ").strip()
                    n_calls = int(n_calls_input)
                    if n_calls >= 5: break
                    else: print("优化尝试次数必须大于或等于 5。")
                except ValueError: print("无效输入，请输入一个数字。")

            try:
                best_params = run_bayesian_optimization(ticker, n_calls=n_calls)
                # 如果优化成功，best_params 会是历史上最好的参数字典
                if best_params:
                     train_now = input("优化完成。是否立即使用找到的最佳参数训练最终模型? (y/n): ").strip().lower()
                     if train_now == 'y':
                         train_final_model(ticker, best_params)
                else:
                     print("优化未能返回有效的最佳参数。")

            except Exception as e:
                print(f"贝叶斯优化过程中发生错误: {e}")
                import traceback
                traceback.print_exc()

        elif choice == '5': # 退出
            print("退出程序。")
            break
        else:
            print("无效选项，请输入 1, 2, 3, 4 或 5。")

if __name__ == "__main__":
    main()
