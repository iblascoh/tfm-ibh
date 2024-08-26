import io
from binance.client import Client
import time
import threading
import pandas as pd
from datetime import datetime
import dash
from dash import dcc, html
import numpy as np
from dash.dependencies import Input, Output, State
from dash import no_update
import plotly.graph_objs as go
import schedule
import json
from io import BytesIO
import boto3
import pandas as pd
import math
import ta.volatility
import warnings
import pickle
import ast
import argparse

warnings.filterwarnings('ignore')



s3 = None
client = None

# Variables globales
BASE_PAIR = 'USDC'
BUCKET = 'data-tfm-iblascoh'
sec_stop = True
symbol_traded = 'LTCUSDC'
hist_data = None
signal = None
orders_key= 'orders.csv'

symbol_hist_lock = threading.Lock()



#lectura y escritura ficheros.

def get_order_book():
    global orders_key
    global s3
    df = pd.read_csv(BytesIO(s3.get_object(Bucket=BUCKET, Key=orders_key)['Body'].read()), delimiter=',', index_col=0, header=0)
    df.reset_index(drop=True, inplace=True)
    df['fills'] = df['fills'].apply(ast.literal_eval)
    df_exploded = df.explode('fills')
    df_normalized = pd.json_normalize(df_exploded['fills'])
    df_normalized.columns = ['f_'+str(col) for col in df_normalized.columns]
    df_final = pd.concat([df.drop(columns=['fills']), df_normalized], axis=1)
    return df_final

def save_order_book(open_order, close_order):
    global orders_key
    global s3
    orders = []
    orders.append(open_order)
    orders.append(close_order)
    df_orders = pd.DataFrame(orders)
    try:
        df = pd.read_csv(BytesIO(s3.get_object(Bucket=BUCKET, Key=orders_key)['Body'].read()))
        df = pd.concat([df, df_orders], axis=0)
        df = df.drop(columns=['Unnamed: 0'])
    except Exception as e:
        df = df_orders
    s3.put_object(Bucket=BUCKET, Key=orders_key, Body=df.to_csv())

   
def format_date(timestamp):
    if type(timestamp) == str:
        return timestamp
    date = datetime.fromtimestamp(timestamp / 1000) # Divide by 1000 to convert to seconds
    formatted_date = date.strftime('%d-%m-%Y %H:%M:%S.%f')
    return formatted_date


def update_hist_data(symbol):
    global s3
    global client
    try:
        df_old = pd.read_parquet(BytesIO(s3.get_object(Bucket=BUCKET, Key='data/'+symbol+'HistData.parquet')['Body'].read()))
        last_date = df_old['openTime'].iloc[-1]
    except:
        df_old = pd.DataFrame(columns=['openTime', 'open', 'high', 'low', 'close', 'volume'])
        last_date = '01-01-2019 00:00:00.000'
    # Define custom start and end time
    start_time = datetime.strptime(last_date, '%d-%m-%Y %H:%M:%S.%f').timestamp()*1000
    end_time = datetime.now().timestamp()*1000
    if (end_time - start_time)/1000/60 >= 15:
        klines = client.get_historical_klines(symbol=symbol, interval='15m', start_str=int(start_time-15000))
        if(len(klines) != 0):
            data = [item[:6] for item in klines]
            df = pd.DataFrame(data, columns=['openTime', 'open', 'high', 'low', 'close', 'volume'])
            df['openTime'] = df['openTime'].apply(lambda x: format_date(x))
            df_old = pd.concat([df_old, df], axis=0)
            df_old.drop_duplicates(subset='openTime', keep='last', inplace=True)
            #Save to s3
            buffer = io.BytesIO()
            df_old.to_parquet(buffer, engine='pyarrow')
            buffer.seek(0)
            s3.put_object(Bucket=BUCKET, Key='data/'+symbol+'HistData.parquet', Body=buffer.getvalue())
        else:
            print('viene vacio')
    #Formatear para entrada
    df_old[["Open", "High", "Low", "Close", "Volume"]] = df_old[["open", "high", "low", "close", "volume"]] .astype(float)  # Convertir todas las columnas a tipo float
    df_old['openTime'] = pd.to_datetime(df_old['openTime'], format='%d-%m-%Y %H:%M:%S.%f')
    df_old.index = pd.DatetimeIndex(df_old['openTime'])
    df_old.set_index('openTime', inplace=True)
    return df_old[["Open", "High", "Low", "Close", "Volume"]].tail(500)

'''
Función para obtener los datos históricos de un símbolo
'''


def truncar(numero, decimales=3):
    factor = 10 ** decimales
    return int(numero * factor) / factor

def run_trader (cliente, symbol_traded):
    trade_thread = threading.Thread(target=execute_trade, args=(client, symbol_traded))
    trade_thread.start()
def schedule_runner():
    while True:
        schedule.run_pending()
        time.sleep(1)
def start_threads(client):
    global symbol_traded
    # TAREA PROGRAMADA PARA EJECUTAR EL TRADER CADA 15 MINUTOS
    schedule.every().hour.at(":59").do(execute_trade, client, symbol_traded)
    schedule.every().hour.at(":14").do(execute_trade, client, symbol_traded)
    schedule.every().hour.at(":29").do(execute_trade, client, symbol_traded)
    schedule.every().hour.at(":44").do(execute_trade, client, symbol_traded)

    # Iniciar el scheduler en un hilo separado
    scheduler_thread = threading.Thread(target=schedule_runner)
    scheduler_thread.start()

'''
Función para obtener el balance de la cuenta del usuario
'''
def get_balances(client):

    account_balances = client.get_account()['balances']
    ticker_info = client.get_all_tickers()
    ticker_prices = {ticker['symbol']: float(ticker['price']) for ticker in ticker_info}

    # Calculate the USDT value of each coin in the user’s account
    coin_values = []
    for coin_balance in account_balances:
    # Get the coin symbol and the free and locked balance of each coin
        coin_symbol = coin_balance['asset']
        unlocked_balance = float(coin_balance['free'])
        locked_balance = float(coin_balance['locked'])

        # If the coin is USDT and the total balance is greater than 1, add it to the list of coins with their USDT values
        if coin_symbol == BASE_PAIR and unlocked_balance + locked_balance > 0.0:
            coin_values.append(unlocked_balance + locked_balance)
        elif unlocked_balance + locked_balance > 0.0:
            if (any(coin_symbol + BASE_PAIR in i for i in ticker_prices)):
                ticker_symbol = coin_symbol + BASE_PAIR
                ticker_price = ticker_prices.get(ticker_symbol)
                coin_usdt_value = (unlocked_balance + locked_balance) * ticker_price
                if coin_usdt_value > 0:
                    coin_values.append(coin_usdt_value)
            else: #SI no es un par
                ticker_symbol = coin_symbol
                ticker_price = ticker_prices.get(ticker_symbol)
                coin_usdt_value = (unlocked_balance + locked_balance) * ticker_price
                if coin_usdt_value > 0:
                    coin_values.append( coin_usdt_value)
    # Return the list of coins and their USDT values
    if len(coin_values) == 0:
        return 0
    return np.array(coin_values).sum()

def get_balance_div(client, symbol_traded):
    account_balances = client.get_account()['balances']
    traded = 0
    base = 0
    for coin_balance in account_balances:
        coin_symbol = coin_balance['asset']
        unlocked_balance = float(coin_balance['free'])
        locked_balance = float(coin_balance['locked'])
        if (unlocked_balance + locked_balance) >= 0:
            if (coin_symbol == symbol_traded.replace(BASE_PAIR, "")):
                print(f"TRADED Coin: {coin_symbol}, Balance: {unlocked_balance + locked_balance}")
                traded = unlocked_balance + locked_balance
            elif coin_symbol == BASE_PAIR: 
                print(f"BASE Coin: {coin_symbol}, Balance: {unlocked_balance + locked_balance}")
                base = unlocked_balance + locked_balance
    return traded, base

def add_features(df):
    df['returns'] = df["Close"].pct_change(1)
    # Día de la semana y hora del día
    df['Day'] = df.index.dayofweek
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
    
    # Indicadores técnicos
    df['Sma_15'] =ta.trend.sma_indicator(df['Close'], window=15, fillna=True)
    df['Sma_10'] =ta.trend.sma_indicator(df['Close'], window=10, fillna=True)
    df['Ema_15'] =ta.trend.ema_indicator(df['Close'], window=15, fillna=True)
    df['Ema_10'] =ta.trend.ema_indicator(df['Close'], window=10, fillna=True)
    df['Rsi_15'] = ta.momentum.rsi(df['Close'], window=15, fillna=True)
    df["Sti_14"] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=14, fillna=True)
    df['Macd'] = ta.trend.macd_diff(df['Close'], window_slow=14, window_fast=7, window_sign=4, fillna=True)
    df['Bollinger_hband'] = ta.volatility.bollinger_hband(df['Close'], window=15, fillna=True)
    df['Bollinger_lband'] = ta.volatility.bollinger_lband(df['Close'], window=15, fillna=True)
    df['Atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=15, fillna=True)
    df['Obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'], fillna=True)
    df['Adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=15, fillna=True)
    df['Cci'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=15, fillna=True)
    df['Dpo'] = ta.trend.dpo(df['Close'], window=15, fillna=True)
    df['Trix'] = ta.trend.trix(df['Close'], window=15, fillna=True)
    df['Buy_signal1'] = np.where((df['Rsi_15'] < 30) & (df['Close'] < df['Bollinger_lband']), 1, 0)
    df['Sell_signal1'] = np.where((df['Rsi_15'] > 70) & (df['Close'] > df['Bollinger_hband']), 1, 0)
    df['Composite_signal'] = df['Sma_15'] + df['Ema_15'] + df['Rsi_15'] + df['Macd'].rolling(window=3).mean()
    df['Buy_signal2'] = np.where(df['Composite_signal'] > 0, 1, 0)
    df['Sell_signal2'] = np.where(df['Composite_signal'] < 0, 1, 0)
    df['Buy_signal3'] = np.where(df['Composite_signal'] > 0, 1, 0)
    df['Sell_signal3'] = np.where(df['Composite_signal'] < 0, 1, 0)
    df['TRIX_signal'] = np.where(df['Trix'] > df['Trix'].rolling(window=10).mean(), 1, 0)
    df['DPO_signal'] = np.where(df['Dpo'] > 0, 1, 0)
    df['CCI_signal'] = np.where(df['Cci'] > 100, 0, np.where(df['Cci'] < -100, 1, 0))
    df['ADX_signal'] = np.where(df['Adx'] > 25, 1, 0)
    # Indicadores sobre precio
    df['Price_change'] = df['Close'].pct_change(1)
    df['Mom'] = df['Close'].pct_change(periods=2)
    for col in df.columns:
        for lag in range(1, 5):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    df.dropna(inplace=True)
    return df

'''
Función para ejecutar una operación de trading
'''
def execute_trade(client, symbol):
    global sec_stop
    global hist_data
    global signal
    
    local_stop = sec_stop
    local_symbol = symbol_traded
    local_hist_data = hist_data
    if (local_stop is  True and local_hist_data is not None and (datetime.now().timestamp() - local_hist_data.index[-1].timestamp())/60 < 15):
        print('enter')
            
        last_ticker = client.get_klines(symbol=local_symbol, interval='1m', limit=1)
        data = [item[:6] for item in last_ticker]
        df = pd.DataFrame(data, columns=['openTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]] .astype(float)
        df['openTime'] = df['openTime'].apply(lambda x: format_date(x))
        df.index = pd.DatetimeIndex(df['openTime'])
        df =pd.concat([local_hist_data, df], axis=0)
        df = add_features(df[["Open", "High", "Low", "Close", "Volume"]])

        limits = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=15, fillna=True).tail(1).values[0]
        #model = pickle.load(open('models/AdaBoostClassifier.pkl', 'rb'))
        model1 = pickle.load(BytesIO(s3.get_object(Bucket=BUCKET, Key='models/AdaBoostClassifier.pkl')['Body'].read()))
        model2 = pickle.load(BytesIO(s3.get_object(Bucket=BUCKET, Key='models/XGBClassifier.pkl')['Body'].read()))
        model3 = pickle.load(BytesIO(s3.get_object(Bucket=BUCKET, Key='models/CatBoostClassifier.pkl')['Body'].read()))

        y_pred1 = model1.predict_proba(df.tail(1))[:, 1]
        y_pred2= model2.predict_proba(df.tail(1))[:, 1]
        y_pred3 = model3.predict_proba(df.tail(1))[:, 1]
        y_pred = np.where((y_pred1 + y_pred2 + y_pred3)/3 > 0.5, 1, 0)[-1]

        signal = 'Buy' if y_pred == 1 else 'Sell'
        take_profit_price = df['Close'].tail(1).iloc[-1] + (2*limits)
        stop_loss_price = df['Close'].tail(1).iloc[-1] - (2*limits)
        #print('signal', signal)
        if signal == 'Buy':
            print(f"Operando en {symbol} hora: {datetime.now()}")
            #Compra
            quantity = obtain_qty(local_symbol, 1)
            try:
                open_order = client.create_order(
                    symbol=local_symbol,
                    side=Client.SIDE_BUY if signal == 'Buy' else None,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                #esperar a que hayan pasado 10 min
                exit =False
                while (exit is False):
                    time.sleep(10)
                    current_price = float(client.get_symbol_ticker(symbol=local_symbol)['price'])
                    if current_price >= take_profit_price or current_price <= stop_loss_price or datetime.now().minute in (13, 28, 43, 58): #Si se pasa el limite o pasa el tiempo vendemos
                        exit = True
                #venta
                quantity = obtain_qty(local_symbol, 0)
                close_order = client.create_order(
                    symbol=local_symbol,
                    side=Client.SIDE_SELL if signal == 'Buy' else None,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                # Guarda la información de ambas órdenes
                save_order_book(open_order, close_order)
                print(f"Cerrando en {symbol} hora: {datetime.now()}")
            except Exception as e:
                print(f"Error al ejecutar la operación: {e}")
        else:
            print('pass')
    else: 
        print("OPERATIVA DETENIDA, NO HAY DATOS RECIENTES")

def obtain_qty(symbol_traded, pos):
    global client
    symbol_info = client.get_symbol_info(symbol_traded)
    # Extraer los filtros de lotSize y otros
    filters = {f['filterType']: f for f in symbol_info['filters']}
    lot_size = filters['LOT_SIZE']
    notional = filters['NOTIONAL']
    min_qty = float(lot_size['minQty'])
    min_not = float(notional['minNotional'])
    step_size = float(lot_size['stepSize'])
    if pos == 1:
        price =  float(client.get_symbol_ticker(symbol=symbol_traded)['price'])
        balance = client.get_asset_balance(asset=BASE_PAIR)
        balance_available = float(balance['free'])
        qty = balance_available / price #Sacamos los ltc a comprar
        if qty >= min_not/price and qty >= min_qty:
            quantity = math.floor(qty / step_size) * step_size
            return quantity
    else:
        price =  float(client.get_symbol_ticker(symbol=symbol_traded)['price'])
        balance = client.get_asset_balance(asset=symbol_traded.replace(BASE_PAIR, ""))
        balance_available = float(balance['free'])
        
        qty = balance_available #los ltc a vender
        if qty*price >= min_not  and qty >= min_qty:
            quantity = math.floor(qty / step_size) * step_size
            return quantity
    
    return 0

def get_stats():
    df_expanded = get_order_book()
    if (len(df_expanded) % 2 != 0):
        df_expanded = df_expanded.drop(df_expanded.tail(1).index)
    n_trades = len(df_expanded)/2
    df = df_expanded
    df['trade_type'] = df['side']  # 'BUY' o 'SELL'

    # Lista para almacenar resultados de trades
    trades = []

    # Variables para seguimiento
    current_trade = None

    # Procesar el DataFrame para identificar trades
    for index, row in df.iterrows():
        if row['trade_type'] == 'BUY':
            current_trade = {
                'buy_price': row['f_price'],
                'buy_qty': row['f_qty'],
                'buy_time': row['transactTime'],
                'buy_orderId': row['orderId']
            }
        elif row['trade_type'] == 'SELL' and current_trade:
            # Si encontramos una venta y hay una compra previa sin cerrar
            current_trade.update({
                'sell_price': row['f_price'],
                'sell_qty': row['f_qty'],
                'sell_time': row['transactTime'],
                'sell_orderId': row['orderId']
            })
            trades.append(current_trade)
            current_trade = None

    # Convertir los trades a un DataFrame
    trades_df = pd.DataFrame(trades)

    trades_df['buy_price'] = trades_df['buy_price'].astype(float)
    trades_df['buy_qty'] = trades_df['buy_qty'].astype(float)
    trades_df['sell_price'] = trades_df['sell_price'].astype(float)
    trades_df['sell_qty'] = trades_df['sell_qty'].astype(float)

    # Calcular rentabilidad y añadir al DataFrame de trades
    trades_df['profit'] = ((trades_df['sell_price'] * trades_df['sell_qty']) - (trades_df['buy_price']* trades_df['buy_qty'])) * trades_df['buy_qty']
    trades_df['result'] = trades_df['profit'] > 0  # True si el trade fue rentable


    # Calcular el porcentaje de aciertos
    accuracy = trades_df['result'].mean() * 100

    # Calcular el retorno medio
    ret_mean = trades_df['profit'].mean()
    # Calcular la curva de equity
    initial_balance = 1
    trades_df['cumulative_profit'] = trades_df['profit'].cumsum()
    trades_df['equity_curve'] = initial_balance + trades_df['cumulative_profit']

    # Calcular el drawdown
    trades_df['equity_max'] = trades_df['equity_curve'].cummax()
    trades_df['drawdown'] = (trades_df['equity_curve'] - trades_df['equity_max']) / trades_df['equity_max']
    trades_df['sell_time'] = pd.to_datetime(trades_df['sell_time'], unit='ms').dt.strftime('%d-%m-%Y %H:%M:%S')
    trades_df.set_index('sell_time', inplace=True)
    return n_trades, accuracy, ret_mean, trades_df

'''
Operativa web
'''
app = dash.Dash(__name__)


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download_button", "n_clicks"),
    prevent_initial_call=True,
)
def descargar_libro(n_clicks):
    try:
        df = get_order_book()
    except Exception as e:
        print(e)
        return no_update
    # Convertir el DataFrame a CSV
    return dcc.send_data_frame(df.to_csv, "libro_de_ordenes.csv", sep=';', index=False)
'''
Funcion para actualizar el simbolo operado y el estado de la operativa
'''
@app.callback(
    Output('operation_status', 'children'),
    Output('selected_symbol', 'children'),
    Input('start_stop_button', 'n_clicks'),
    Input('symbol_selector', 'value'),
    State('operation_status', 'children'),
    State('selected_symbol', 'children')
)
def update_operation(n_clicks, selected_symbol, current_status, current_symbol):
    global sec_stop
    global symbol_traded
    if n_clicks % 2 == 1:
        sec_stop = not sec_stop
    symbol_traded = selected_symbol
    status_text = f"Estado: {'On' if sec_stop else 'Off'}"
    symbol_text = f"Simbolo Operado: {symbol_traded}"
    return status_text, symbol_text

'''
Funcion para actualizar el grafico de velas
'''
@app.callback(
    Output('symbol_price_chart', 'figure'),
    Input('symbol_selector', 'value'),
    Input('interval-component', 'n_intervals'),
    State('symbol_selector', 'value')
)
def update_symbol(symbol, n, prev_simbol):
    global hist_data
    if  datetime.now().minute not in  (1, 16, 31, 46)   and prev_simbol == symbol and n!= 0:
        return no_update
    with symbol_hist_lock:
        hist_data = None
        df = update_hist_data(symbol)
        hist_data = df.tail(30)    
        fig_symbol = go.Figure(data=[go.Candlestick(x=df.index,
                                                open=df['Open'],
                                                high=df['High'],
                                                low=df['Low'],
                                                close=df['Close'])])
        fig_symbol.update_layout(title={'text': f'Velas {symbol} ','x': 0.5,'xanchor': 'center','font': {'color': 'white'}},
                                    xaxis_title='Fecha',
                                    yaxis_title='Precio ($)',
                                    plot_bgcolor='#2a2a3b', 
                                    paper_bgcolor='#1e1e2f',
                                    xaxis_title_font=dict(color='white'), 
                                    yaxis_title_font=dict(color='white'),
                                    xaxis=dict(
                                        tickfont=dict(color='white'), 
                                        titlefont=dict(color='white') 
                                    ),
                                    yaxis=dict(
                                        tickfont=dict(color='white'),
                                        titlefont=dict(color='white')  
                                    )
                                )
    return fig_symbol

'''
Funcion para actualizar el balance, n_trades, prob_acierto, return y pred_actual
'''
@app.callback(
    
    Output('balance_pie_chart', 'figure'),
    Output('equity_curve', 'figure'),
    Output('balance', 'children'),
    Output('n_trades', 'children'),
    Output('ret_mean', 'children'),
    Output('prob_acierto', 'children'),
    Output('pred_actual', 'children'),
    Input('interval-component', 'n_intervals')
    )
def update_equity_drawdown(n):
    global signal
    global client
    if  datetime.now().minute not in  (1, 16, 31, 46) and n!=0:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update
    else:
        #Balance
        balances = f"Balance actual: {round(get_balances(client), 3)}$"        #pred_actual
        if (signal is not None):
            pred_actual = f"Prediccion actual: {signal}"
        else: 
            pred_actual = f"Prediccion actual: N/A"
        try:
            n_trades, accuracy, mean_ret, trades_df = get_stats()
        except Exception as e:
            print(e)
            return no_update, no_update, balances, no_update, no_update, no_update, pred_actual
        #n_trades desde fichero
        n_trades = f"Nº Trades: {n_trades}"
        #mean_ret desde fichero
        ret_mean = f"Retorno medio: {round(mean_ret, 3)}%"
        #prob_acierto desde fichero
        accuracy = f"% Acierto: {accuracy}"
        #return desde fichero

        # plot evolucion balances desde fichero
        equity_curve = go.Figure(data=[go.Scatter(x=trades_df.index, y=trades_df['equity_curve'], mode='markers+lines', name='Curva de P&L', line=dict(color='red'), marker=dict(size=10))])
        equity_curve.update_layout(title={'text': f'P&L','x': 0.5,'xanchor': 'center','font': {'color': 'white'}},
                                    xaxis_title='Fecha',
                                    yaxis_title='%',
                                    plot_bgcolor='#2a2a3b', 
                                    paper_bgcolor='#1e1e2f',
                                    xaxis_title_font=dict(color='white'), 
                                    yaxis_title_font=dict(color='white'),
                                    xaxis=dict(
                                        tickfont=dict(color='white'), 
                                        titlefont=dict(color='white') 
                                    ),
                                    yaxis=dict(
                                        tickfont=dict(color='white'),
                                        titlefont=dict(color='white')  
                                    )
                                )
        balance_pie_chart = go.Figure(data=[go.Scatter(x=trades_df.index, y=trades_df['drawdown'], mode='markers+lines', name='Drawdown', line=dict(color='red'), marker=dict(size=10))])
        balance_pie_chart.update_layout(title={'text': f'Drawdown %','x': 0.5,'xanchor': 'center','font': {'color': 'white'}},
                                    xaxis_title='Fecha',
                                    yaxis_title='%',
                                    plot_bgcolor='#2a2a3b', 
                                    paper_bgcolor='#1e1e2f',
                                    xaxis_title_font=dict(color='white'), 
                                    yaxis_title_font=dict(color='white'),
                                    xaxis=dict(
                                        tickfont=dict(color='white'), 
                                        titlefont=dict(color='white') 
                                    ),
                                    yaxis=dict(
                                        tickfont=dict(color='white'),
                                        titlefont=dict(color='white')  
                                    )
                                )

    return equity_curve, balance_pie_chart, balances, n_trades, ret_mean, accuracy, pred_actual



def config(bk, bs, ak, ack):
    global client
    global s3
 
    client = Client(bk, bs)

    s3 = boto3.client(
        's3', 
        aws_access_key_id=ak,        # Reemplaza con tu clave de acceso
        aws_secret_access_key=ack,    # Reemplaza con tu clave secreta
        region_name='ap-northeast-1'                # Asegúrate de que coincida con tu región
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script para aceptar variables globales como argumentos.")
    parser.add_argument('--api_key', type=str, required=True, help="La clave API.")
    parser.add_argument('--api_secret', type=str, required=True, help="El secreto de la API.")
    parser.add_argument('--aws_client', type=str, required=True, help="El cliente AWS.")
    parser.add_argument('--aws_key', type=str, required=True, help="La clave AWS.")
    args = parser.parse_args()
    config(args.api_key, args.api_secret, args.aws_client, args.aws_key)
    start_threads(client)

    app.layout = html.Div(
    style={'background-color': '#1e1e2f', 'color': 'white', 'font-family': 'Arial, sans-serif', 'padding': '20px'},
    children=[
        html.H1("PANEL DE CONTROL DE OPERATIVA", style={'text-align': 'center', 'font-size': '32px', 'color': '#ffd700', 'margin-bottom': '10px'}),
        html.Div([
            html.Div([
                html.P('Descargar libro de órdenes', style={
                    'background-color': '#2a2a3b', 'color': 'white', 'font-family': 'Arial, sans-serif', 'textAlign': 'center',
                }),
                html.Button(
                    'Descargar', 
                    id='download_button', 
                    n_clicks=0, 
                    style={'background-color': '#ffd700', 'color': '#2a2a3b', 'padding': '10px', 'border-radius': '5px', 'cursor': 'pointer', 'width': '100%'}
                ),
                dcc.Download(id="download-dataframe-csv")
            ], style={
                'border': '1px solid #ffd700',
                'background-color': '#2a2a3b', 
                'color': '#2a2a3b',
                'text': '#2a2a3b',
                'padding': '5px', 
                'border-radius': '5px',
                'margin-top': '5px',
                'flex': '1'  # Flex grow
            }),
            html.Div([
                html.P('Control', style={
                    'background-color': '#2a2a3b', 'color': 'white', 'font-family': 'Arial, sans-serif','textAlign': 'center',
                }),
                html.Button(
                    'Start/Stop', 
                    id='start_stop_button', 
                    n_clicks=0, 
                    style={'background-color': '#ffd700', 'color': '#2a2a3b', 'padding': '10px', 'border-radius': '5px', 'cursor': 'pointer', 'width': '100%'}
                )
            ], style={
                'border': '1px solid #ffd700',
                'background-color': '#2a2a3b', 
                'color': '#2a2a3b',
                'text': '#2a2a3b',
                'padding': '5px', 
                'border-radius': '5px',
                'margin-top': '5px',
                'flex': '1'  # Flex grow
            }),
            html.Div([
                html.P('Selección del símbolo', style={
                    'background-color': '#2a2a3b', 'color': 'white', 'font-family': 'Arial, sans-serif','textAlign': 'center',
                }),
                dcc.Dropdown(
                    id='symbol_selector',
                    options=[
                    {'label': s['symbol'], 'value': s['symbol']}
                    for s in client.get_exchange_info()['symbols']
                    if s['status'] == 'TRADING' and s['symbol'].upper().endswith(BASE_PAIR)
                ],
                    value=symbol_traded,
                    style={
                        'border': '1px solid #ffd700',
                        'background-color': '#ffd700', 
                        'border-radius': '5px',
                        'color': '#2a2a3b',  # Adjusted to be visible on dark background
                    }
                )
            ], style={
                'border': '1px solid #ffd700',
                'background-color': '#2a2a3b', 
                'color': '#2a2a3b',
                'text': '#2a2a3b',
                'padding': '5px', 
                'border-radius': '5px',
                'margin-top': '5px',
                'flex': '1'  # Flex grow
            })
        ], style={
            'display': 'flex',
            'justify-content': 'space-between'  # Distribute space between columns
        }),
        
        html.Div(id='operation_status', style={'border': '1px solid #ffd700','margin-top': '20px', 'padding': '10px', 'background-color': '#2a2a3b', 'border-radius': '5px'}),
        
        html.Div(id='selected_symbol', style={'border': '1px solid #ffd700','margin-top': '10px', 'padding': '10px', 'background-color': '#2a2a3b', 'border-radius': '5px'}),
        
        dcc.Graph(id='symbol_price_chart',
                  figure =go.Figure().update_layout(title={'text': f'Velas ','x': 0.5,'xanchor': 'center','font': {'color': 'white'}},
                                    xaxis_title='Fecha',
                                    yaxis_title='Precio ($)',
                                    plot_bgcolor='#2a2a3b', 
                                    paper_bgcolor='#1e1e2f',
                                    xaxis_title_font=dict(color='white'), 
                                    yaxis_title_font=dict(color='white'),
                                    xaxis=dict(
                                        tickfont=dict(color='white'), 
                                        titlefont=dict(color='white') 
                                    ),
                                    yaxis=dict(
                                        tickfont=dict(color='white'),
                                        titlefont=dict(color='white')  
                                    )
                                ),
                   style={'background-color': '#2a2a3b', 'margin-top': '20px', 'border-radius': '5px'}),
        
        html.Table(
            style={'border': '1px solid #ffd700','width': '100%', 'margin-top': '20px', 'background-color': '#2a2a3b', 'padding': '10px', 'border-spacing': '10px', 'border-radius': '5px'},
            children=[
                html.Tr(
                    children=[
                        html.Td(html.Div("Balance", id='balance', style={'padding': '10px', 'background-color': '#3a3a4b', 'border-radius': '5px'})),
                        html.Td(html.Div("N Trades", id='n_trades', style={'padding': '10px', 'background-color': '#3a3a4b', 'border-radius': '5px'})),
                        html.Td(html.Div("Retorno medio", id='ret_mean', style={'padding': '10px', 'background-color': '#3a3a4b', 'border-radius': '5px'})),
                        html.Td(html.Div("Probabilidad Acierto", id='prob_acierto', style={'padding': '10px', 'background-color': '#3a3a4b', 'border-radius': '5px'})),
                        html.Td(html.Div("Prediccion Actual", id='pred_actual', style={'padding': '10px', 'background-color': '#3a3a4b', 'border-radius': '5px'})),
                    ]
                )
            ]
        ),
        
        html.Div(
            style={'display': 'flex', 'margin-top': '10px'},
            children=[
                html.Div(style={'width': '50%', 'background-color': '#2a2a3b'},
                          children=[dcc.Graph(id='balance_pie_chart',
                                               figure=go.Figure().update_layout(title={'text': f'P&L','x': 0.5,'xanchor': 'center','font': {'color': 'white'}},
                                    xaxis_title='Fecha',
                                    yaxis_title='Valor ($)',
                                    plot_bgcolor='#2a2a3b', 
                                    paper_bgcolor='#1e1e2f',
                                    xaxis_title_font=dict(color='white'), 
                                    yaxis_title_font=dict(color='white'),
                                    xaxis=dict(
                                        tickfont=dict(color='white'), 
                                        titlefont=dict(color='white') 
                                    ),
                                    yaxis=dict(
                                        tickfont=dict(color='white'),
                                        titlefont=dict(color='white')  
                                    )
                                ))]),
                html.Div(style={'width': '50%', 'background-color': '#2a2a3b' },
                          children=[dcc.Graph(id='equity_curve',
                                    figure=go.Figure().update_layout(title={'text': f'Drawdown %','x': 0.5,'xanchor': 'center','font': {'color': 'white'}},
                                    xaxis_title='Fecha',
                                    yaxis_title='%',
                                    plot_bgcolor='#2a2a3b', 
                                    paper_bgcolor='#1e1e2f',
                                    xaxis_title_font=dict(color='white'), 
                                    yaxis_title_font=dict(color='white'),
                                    xaxis=dict(
                                        tickfont=dict(color='white'), 
                                        titlefont=dict(color='white') 
                                    ),
                                    yaxis=dict(
                                        tickfont=dict(color='white'),
                                        titlefont=dict(color='white')  
                                    )
                                )
                           )]),
            ]
        ),
        html.Div(
            style={'text-align': 'center', 'font-size': '14px', 'color': '#ffd700', 'margin-top': '10px'},
            children=["Ignacio Blasco Hernandes TFM MUIT 2024"]
        ),
        dcc.Interval(
            id='interval-component',
            interval = 60 * 1000,  # 5 minutos en milisegundos
            n_intervals=0
        )
    ]
)
    
    app.run(debug=False, host='0.0.0.0', port=8050)