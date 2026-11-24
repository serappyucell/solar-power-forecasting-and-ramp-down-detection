from flask import Flask, render_template, request, redirect, url_for, flash  # type: ignore
import pandas as pd  # type: ignore
import os
from werkzeug.utils import secure_filename  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
import uuid
import traceback
import numpy as np  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # type: ignore
from sklearn.linear_model import LinearRegression, SGDRegressor  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.neighbors import KNeighborsRegressor  # type: ignore
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error  # type: ignore
import xgboost as xgb  # type: ignore

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Kullanıcıya mesaj göstermek için secret key
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls'}

# Klasör yoksa oluştur
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def perform_eda(df):
    try:
        eda_results = {}
        
        # Temel istatistikler
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        basic_stats = df[numeric_columns].describe()
        eda_results['basic_stats'] = basic_stats.to_html(classes='min-w-full divide-y divide-gray-200 table-auto',
                                                       float_format=lambda x: '{:.2f}'.format(x))

        # Veri dağılımı analizi
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        sns.boxplot(data=df['Üretim (Veriş)'])
        plt.title('Üretim Değerleri Dağılımı (Box Plot)')
        
        plt.subplot(2, 1, 2)
        sns.histplot(data=df['Üretim (Veriş)'], bins=50)
        plt.title('Üretim Değerleri Dağılımı (Histogram)')
        
        plt.tight_layout()
        os.makedirs('static', exist_ok=True)
        dist_plot_filename = f"eda_dist_{uuid.uuid4().hex}.png"
        plt.savefig(os.path.join('static', dist_plot_filename))
        plt.close()
        eda_results['distribution_plot'] = url_for('static', filename=dist_plot_filename)

        # Üretim değerlerinin saatlik ortalaması
        df['Saat'] = df['Profil Tarihi'].dt.hour
        hourly_mean = df.groupby('Saat')['Üretim (Veriş)'].mean()
        
        plt.figure(figsize=(12, 6))
        hourly_mean.plot(kind='bar')
        plt.title('Saatlik Ortalama Üretim')
        plt.xlabel('Saat')
        plt.ylabel('Ortalama Üretim')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        hourly_plot_filename = f"eda_hourly_{uuid.uuid4().hex}.png"
        plt.savefig(os.path.join('static', hourly_plot_filename))
        plt.close()
        eda_results['hourly_plot'] = url_for('static', filename=hourly_plot_filename)

        # Aylık analiz
        df['Ay'] = df['Profil Tarihi'].dt.month
        monthly_mean = df.groupby('Ay')['Üretim (Veriş)'].mean()
        
        plt.figure(figsize=(12, 6))
        monthly_mean.plot(kind='bar')
        plt.title('Aylık Ortalama Üretim')
        plt.xlabel('Ay')
        plt.ylabel('Ortalama Üretim')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        monthly_plot_filename = f"eda_monthly_{uuid.uuid4().hex}.png"
        plt.savefig(os.path.join('static', monthly_plot_filename))
        plt.close()
        eda_results['monthly_plot'] = url_for('static', filename=monthly_plot_filename)

        return eda_results

    except Exception as e:
        print(f"EDA sırasında hata: {str(e)}")
        return None

def train_random_forest(df):
    try:
        # Veri tipi dönüşümleri ve kontroller
        df['Profil Tarihi'] = pd.to_datetime(df['Profil Tarihi'], dayfirst=True)
        df = df.sort_values('Profil Tarihi')  # Zaman sırasına göre sırala
        df = df.reset_index(drop=True)

        # Lag feature'lar oluştur
        lag_steps = [1, 2, 3]  # Geçmiş 1, 2 ve 3 saatlik değerler kullanılacak
        for lag in lag_steps:
            df[f'lag_{lag}'] = df['Üretim (Veriş)'].shift(lag)
        
        # İlk lag'lerden dolayı oluşan boş satırları düşür
        df = df.dropna()

        # Giriş (X) ve çıkış (y) verilerini ayır
        X = df[[f'lag_{lag}' for lag in lag_steps]]
        y = df['Üretim (Veriş)']

        # Eğitim ve test verilerini ayır
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        # Modeli oluştur ve eğit
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Tahmin yap
        y_pred = model.predict(X_test)

        # Başarıyı değerlendir
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Random Forest sonuç grafiği
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values, label='Gerçek Değerler', alpha=0.7)
        plt.plot(y_pred, label='Tahmin Edilen Değerler', alpha=0.7)
        plt.legend()
        plt.title('Gerçek ve Tahmin Edilen Üretim Değerleri')
        plt.xlabel('Zaman')
        plt.ylabel('Üretim (Veriş)')
        plt.grid(True, alpha=0.3)
        
        # Görseli kaydet
        os.makedirs('static', exist_ok=True)
        rf_plot_filename = f"rf_{uuid.uuid4().hex}.png"
        rf_plot_path = os.path.join('static', rf_plot_filename)
        plt.savefig(rf_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return {
            'mse': round(mse, 2),
            'r2': round(r2, 4),
            'plot_url': url_for('static', filename=rf_plot_filename)
        }
    except Exception as e:
        print(f"Random Forest eğitimi sırasında hata: {str(e)}")
        return None

def train_linear_regression(df):
    try:
        # Veri tipi dönüşümleri ve kontroller
        df['Profil Tarihi'] = pd.to_datetime(df['Profil Tarihi'], dayfirst=True)
        df = df.sort_values('Profil Tarihi')  # Zaman sırasına göre sırala
        df = df.reset_index(drop=True)

        # Lag feature'lar oluştur
        lag_steps = [1, 2, 3]  # Geçmiş 1, 2 ve 3 saatlik değerler kullanılacak
        for lag in lag_steps:
            df[f'lag_{lag}'] = df['Üretim (Veriş)'].shift(lag)
        
        # İlk lag'lerden dolayı oluşan boş satırları düşür
        df = df.dropna()

        # Giriş (X) ve çıkış (y) verilerini ayır
        X = df[[f'lag_{lag}' for lag in lag_steps]]
        y = df['Üretim (Veriş)']

        # Eğitim ve test verilerini ayır
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        # Modeli oluştur ve eğit
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Tahmin yap
        y_pred = model.predict(X_test)

        # Başarıyı değerlendir
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Linear Regression sonuç grafiği
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values, label='Gerçek Değerler', alpha=0.7)
        plt.plot(y_pred, label='Tahmin Edilen Değerler', alpha=0.7)
        plt.legend()
        plt.title('Gerçek ve Tahmin Edilen Üretim Değerleri')
        plt.xlabel('Zaman')
        plt.ylabel('Üretim (Veriş)')
        plt.grid(True, alpha=0.3)
        
        # Görseli kaydet
        os.makedirs('static', exist_ok=True)
        lr_plot_filename = f"lr_{uuid.uuid4().hex}.png"
        lr_plot_path = os.path.join('static', lr_plot_filename)
        plt.savefig(lr_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return {
            'mae': round(mae, 2),
            'mape': round(mape, 2),
            'mse': round(mse, 2),
            'rmse': round(rmse, 2),
            'r2': round(r2, 4),
            'plot_url': url_for('static', filename=lr_plot_filename)
        }
    except Exception as e:
        print(f"Linear Regression eğitimi sırasında hata: {str(e)}")
        return None

def train_gradient_boosting(df):
    try:
        # Veri tipi dönüşümleri ve kontroller
        df['Profil Tarihi'] = pd.to_datetime(df['Profil Tarihi'], dayfirst=True)
        df = df.sort_values('Profil Tarihi')  # Zaman sırasına göre sırala
        df = df.reset_index(drop=True)

        # Lag feature'lar oluştur
        lag_steps = [1, 2, 3]  # Geçmiş 1, 2 ve 3 saatlik değerler kullanılacak
        for lag in lag_steps:
            df[f'lag_{lag}'] = df['Üretim (Veriş)'].shift(lag)
        
        # İlk lag'lerden dolayı oluşan boş satırları düşür
        df = df.dropna()

        # Giriş (X) ve çıkış (y) verilerini ayır
        X = df[[f'lag_{lag}' for lag in lag_steps]]
        y = df['Üretim (Veriş)']

        # Eğitim ve test verilerini ayır
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        # Modeli oluştur ve eğit
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Tahmin yap
        y_pred = model.predict(X_test)

        # Başarıyı değerlendir
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Gradient Boosting sonuç grafiği
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values, label='Gerçek Değerler', alpha=0.7)
        plt.plot(y_pred, label='Tahmin Edilen Değerler', alpha=0.7)
        plt.legend()
        plt.title('Gerçek ve Tahmin Edilen Üretim Değerleri')
        plt.xlabel('Zaman')
        plt.ylabel('Üretim (Veriş)')
        plt.grid(True, alpha=0.3)
        
        # Görseli kaydet
        os.makedirs('static', exist_ok=True)
        gbr_plot_filename = f"gbr_{uuid.uuid4().hex}.png"
        gbr_plot_path = os.path.join('static', gbr_plot_filename)
        plt.savefig(gbr_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return {
            'mae': round(mae, 2),
            'mape': round(mape, 2),
            'mse': round(mse, 2),
            'rmse': round(rmse, 2),
            'r2': round(r2, 4),
            'plot_url': url_for('static', filename=gbr_plot_filename)
        }
    except Exception as e:
        print(f"Gradient Boosting eğitimi sırasında hata: {str(e)}")
        return None

def train_xgboost(df):
    try:
        # Veri tipi dönüşümleri ve kontroller
        df['Profil Tarihi'] = pd.to_datetime(df['Profil Tarihi'], dayfirst=True)
        df = df.sort_values('Profil Tarihi')  # Zaman sırasına göre sırala
        df = df.reset_index(drop=True)

        # Lag feature'lar oluştur
        lag_steps = [1, 2, 3]  # Geçmiş 1, 2 ve 3 saatlik değerler kullanılacak
        for lag in lag_steps:
            df[f'lag_{lag}'] = df['Üretim (Veriş)'].shift(lag)
        
        # İlk lag'lerden dolayı oluşan boş satırları düşür
        df = df.dropna()

        # Giriş (X) ve çıkış (y) verilerini ayır
        X = df[[f'lag_{lag}' for lag in lag_steps]]
        y = df['Üretim (Veriş)']

        # Eğitim ve test verilerini ayır
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        # Modeli oluştur ve eğit
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Tahmin yap
        y_pred = model.predict(X_test)

        # Başarıyı değerlendir
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # XGBoost sonuç grafiği
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values, label='Gerçek Değerler', alpha=0.7)
        plt.plot(y_pred, label='Tahmin Edilen Değerler', alpha=0.7)
        plt.legend()
        plt.title('Gerçek ve Tahmin Edilen Üretim Değerleri')
        plt.xlabel('Zaman')
        plt.ylabel('Üretim (Veriş)')
        plt.grid(True, alpha=0.3)
        
        # Görseli kaydet
        os.makedirs('static', exist_ok=True)
        xgb_plot_filename = f"xgb_{uuid.uuid4().hex}.png"
        xgb_plot_path = os.path.join('static', xgb_plot_filename)
        plt.savefig(xgb_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return {
            'mae': round(mae, 2),
            'mape': round(mape, 2),
            'mse': round(mse, 2),
            'rmse': round(rmse, 2),
            'r2': round(r2, 4),
            'plot_url': url_for('static', filename=xgb_plot_filename)
        }
    except Exception as e:
        print(f"XGBoost eğitimi sırasında hata: {str(e)}")
        return None

def train_knn(df):
    try:
        # Veri tipi dönüşümleri ve kontroller
        df['Profil Tarihi'] = pd.to_datetime(df['Profil Tarihi'], dayfirst=True)
        df = df.sort_values('Profil Tarihi')  # Zaman sırasına göre sırala
        df = df.reset_index(drop=True)

        # Lag feature'lar oluştur
        lag_steps = [1, 2, 3]  # Geçmiş 1, 2 ve 3 saatlik değerler kullanılacak
        for lag in lag_steps:
            df[f'lag_{lag}'] = df['Üretim (Veriş)'].shift(lag)
        
        # İlk lag'lerden dolayı oluşan boş satırları düşür
        df = df.dropna()

        # Giriş (X) ve çıkış (y) verilerini ayır
        X = df[[f'lag_{lag}' for lag in lag_steps]]
        y = df['Üretim (Veriş)']

        # Eğitim ve test verilerini ayır
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        # Modeli oluştur ve eğit
        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(X_train, y_train)

        # Tahmin yap
        y_pred = model.predict(X_test)

        # Başarıyı değerlendir
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # KNN sonuç grafiği
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values, label='Gerçek Değerler', alpha=0.7)
        plt.plot(y_pred, label='Tahmin Edilen Değerler', alpha=0.7)
        plt.legend()
        plt.title('KNN - Gerçek ve Tahmin Edilen Üretim Değerleri')
        plt.xlabel('Zaman')
        plt.ylabel('Üretim (Veriş)')
        plt.grid(True, alpha=0.3)
        
        # Görseli kaydet
        os.makedirs('static', exist_ok=True)
        knn_plot_filename = f"knn_{uuid.uuid4().hex}.png"
        knn_plot_path = os.path.join('static', knn_plot_filename)
        plt.savefig(knn_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return {
            'mae': round(mae, 2),
            'mape': round(mape, 2),
            'mse': round(mse, 2),
            'rmse': round(rmse, 2),
            'r2': round(r2, 4),
            'plot_url': url_for('static', filename=knn_plot_filename)
        }
    except Exception as e:
        print(f"KNN eğitimi sırasında hata: {str(e)}")
        return None

def train_sgd(df):
    try:
        # Veri tipi dönüşümleri ve kontroller
        df['Profil Tarihi'] = pd.to_datetime(df['Profil Tarihi'], dayfirst=True)
        df = df.sort_values('Profil Tarihi')  # Zaman sırasına göre sırala
        df = df.reset_index(drop=True)

        # Lag feature'lar oluştur
        lag_steps = [1, 2, 3]  # Geçmiş 1, 2 ve 3 saatlik değerler kullanılacak
        for lag in lag_steps:
            df[f'lag_{lag}'] = df['Üretim (Veriş)'].shift(lag)
        
        # İlk lag'lerden dolayı oluşan boş satırları düşür
        df = df.dropna()

        # Giriş (X) ve çıkış (y) verilerini ayır
        X = df[[f'lag_{lag}' for lag in lag_steps]]
        y = df['Üretim (Veriş)']

        # Eğitim ve test verilerini ayır
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        # Veriyi ölçekle
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Modeli oluştur ve eğit
        model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Tahmin yap
        y_pred = model.predict(X_test_scaled)

        # Başarıyı değerlendir
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # SGD sonuç grafiği
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values, label='Gerçek Değerler', alpha=0.7)
        plt.plot(y_pred, label='Tahmin Edilen Değerler', alpha=0.7)
        plt.legend()
        plt.title('SGD - Gerçek ve Tahmin Edilen Üretim Değerleri')
        plt.xlabel('Zaman')
        plt.ylabel('Üretim (Veriş)')
        plt.grid(True, alpha=0.3)
        
        # Görseli kaydet
        os.makedirs('static', exist_ok=True)
        sgd_plot_filename = f"sgd_{uuid.uuid4().hex}.png"
        sgd_plot_path = os.path.join('static', sgd_plot_filename)
        plt.savefig(sgd_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return {
            'mae': round(mae, 2),
            'mape': round(mape, 2),
            'mse': round(mse, 2),
            'rmse': round(rmse, 2),
            'r2': round(r2, 4),
            'plot_url': url_for('static', filename=sgd_plot_filename)
        }
    except Exception as e:
        print(f"SGD eğitimi sırasında hata: {str(e)}")
        return None


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    algorithm = request.form.get('algorithm')
    
    if file.filename == '':
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Dosya uzantısına göre oku
            file_extension = filename.rsplit('.', 1)[1].lower()
            if file_extension == 'csv':
                # Utf-8 ile deneyin, olmazsa latin1 ile okumayı deneyin
                try:
                    df = pd.read_csv(filepath, encoding='utf-8')
                except:
                    df = pd.read_csv(filepath, encoding='latin1')
            elif file_extension in ['xlsx', 'xls']:
                # Excel dosyasını okurken engine olarak openpyxl kullan
                df = pd.read_excel(filepath, engine='openpyxl')
            
            print(f"Sütunlar: {df.columns.tolist()}")
            print(f"Veri tipleri: {df.dtypes}")
            
            # İlk olarak tarih sütununu dönüştür
            if 'Profil Tarihi' in df.columns:
                df['Profil Tarihi'] = pd.to_datetime(df['Profil Tarihi'], errors='coerce')
                print(f"Tarih sütunu dönüştürüldü: {df['Profil Tarihi'].dtype}")

            # Sonra üretim sütununu dönüştür
            if 'Üretim (Veriş)' in df.columns:
                # Virgül içeren sayıları noktaya dönüştür (ör: 1,5 -> 1.5)
                if df['Üretim (Veriş)'].dtype == 'object' and isinstance(df['Üretim (Veriş)'].iloc[0], str):
                    df['Üretim (Veriş)'] = df['Üretim (Veriş)'].str.replace(',', '.', regex=False)
                df['Üretim (Veriş)'] = pd.to_numeric(df['Üretim (Veriş)'], errors='coerce')
                print(f"Üretim sütunu dönüştürüldü: {df['Üretim (Veriş)'].dtype}")
            
            # Sayısal sütunları seç
            numeric_df = df.select_dtypes(include=['number'])
            numeric_df = numeric_df.dropna(axis=1, how='all')
            print(f"Sayısal sütunlar: {numeric_df.columns.tolist()}")

            # Grafik oluştur
            plot_url = None
            try:
                # NaN değerleri olan satırları çıkar
                grafik_df = df.dropna(subset=['Profil Tarihi', 'Üretim (Veriş)'])
                print(f"Grafik için kullanılabilir veri sayısı: {len(grafik_df)}")

                if not grafik_df.empty:
                    plt.figure(figsize=(10, 6))
                    plt.plot(grafik_df['Profil Tarihi'], grafik_df['Üretim (Veriş)'],
                            marker='.', linestyle='-', color='seagreen', markersize=2)
                    plt.xlabel('Tarih')
                    plt.ylabel('Üretim')
                    plt.title('Üretim Değerleri')
                    plt.xticks(rotation=45)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()

                    # Görseli kaydet
                    os.makedirs('static', exist_ok=True)
                    plot_filename = f"{uuid.uuid4().hex}.png"
                    plot_path = os.path.join('static', plot_filename)
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()

                    plot_url = url_for('static', filename=plot_filename)
                else:
                    print("Grafik için kullanılabilir veri bulunamadı!")
            except Exception as e:
                print(f"Grafik oluşturulurken hata oluştu: {e}")

            # EDA analizi yap
            eda_results = perform_eda(df.copy())

            # Algoritma seçimine göre modeli eğit
            prediction_results = None
            if algorithm == 'random_forest':
                prediction_results = train_random_forest(df.copy())
            elif algorithm == 'linear_regression':
                prediction_results = train_linear_regression(df.copy())
            elif algorithm == 'gradient_boosting':
                prediction_results = train_gradient_boosting(df.copy())
            elif algorithm == 'xgboost':
                prediction_results = train_xgboost(df.copy())
            elif algorithm == 'knn':
                prediction_results = train_knn(df.copy())
            elif algorithm == 'sgd':
                prediction_results = train_sgd(df.copy())

            # Veri setinde hiç sayısal veri yoksa
            if numeric_df.empty:
                print("HATA: Sayısal veri bulunamadı!")
                # Tüm veriyi olduğu gibi göster
                table_html = df.head(100).to_html(classes='min-w-full divide-y divide-gray-200 table-auto', 
                                           index=False, 
                                           border=0)
                
                data_info = {
                    'rows': df.shape[0],
                    'columns': df.shape[1],
                    'column_names': df.columns.tolist()
                }
                
                return render_template('index.html', error="Dosyada otomatik tespit edilen sayısal veri bulunamadı. Ancak tüm veriler aşağıda gösteriliyor.", 
                                      table_html=table_html, data_info=data_info, filename=filename)
            
            stats = {
                'min': numeric_df.min().min(),
                'max': numeric_df.max().max(),
                'mean': round(numeric_df.mean().mean(), 2)
            }

            # Veri tablosu oluştur - HTML'e dönüştür
            # İlk 100 satır ile sınırla (performans için)
            table_html = df.head(100).to_html(classes='min-w-full divide-y divide-gray-200 table-auto', 
                                           index=False, 
                                           border=0)
            
            # DataFrame yapısı bilgisini ekle
            data_info = {
                'rows': df.shape[0],
                'columns': df.shape[1],
                'column_names': df.columns.tolist()
            }

            return render_template('index.html', 
                               stats=stats, 
                               prediction=prediction_results,
                               plot_url=plot_url, 
                               eda_results=eda_results,
                               table_html=table_html, 
                               data_info=data_info, 
                               filename=filename, 
                               algorithm=algorithm)
        
        except Exception as e:
            error_message = f"Dosya işlenirken bir hata oluştu: {str(e)}"
            print(error_message)
            print(traceback.format_exc())
            return render_template('index.html', error=error_message)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)