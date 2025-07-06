#!/usr/bin/env python3
"""
src/train.py - Script de treinamento baseado no seu notebook

Este arquivo é executado DENTRO do SageMaker Training Job.
Recebe dados já processados e treina o modelo usando sua lógica do notebook.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML libraries (exatamente como no seu notebook)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, 
    classification_report, 
    precision_score,
    recall_score,
    f1_score
)

def parse_args():
    """Parse hyperparameters do SageMaker"""
    parser = argparse.ArgumentParser()
    
    # Hyperparameters (baseado no seu notebook)
    parser.add_argument('--C', type=float, default=0.1)  # Regularização
    parser.add_argument('--penalty', type=str, default='l2')
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--random_state', type=int, default=42)
    
    # SageMaker paths (automáticos)
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    
    return parser.parse_args()

def load_processed_data(data_dir):
    """
    Carrega dados já processados (vem da pipeline.py)
    """
    print("📊 Carregando dados processados...")
    
    files = os.listdir(data_dir)
    csv_files = [f for f in files if f.endswith('.csv')]
    
    if not csv_files:
        raise ValueError("❌ Nenhum arquivo CSV encontrado")
    
    # Carregar primeiro arquivo CSV
    data_file = os.path.join(data_dir, csv_files[0])
    data = pd.read_csv(data_file)
    
    print(f"✅ Dados carregados: {data.shape}")
    print(f"📋 Colunas: {list(data.columns)}")
    
    # Verificar target
    if 'SeriousDlqin2yrs' not in data.columns:
        raise ValueError("❌ Target 'SeriousDlqin2yrs' não encontrado")
    
    # Separar features e target
    X = data.drop('SeriousDlqin2yrs', axis=1)
    y = data['SeriousDlqin2yrs']
    
    print(f"📊 Features: {X.shape}")
    print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
    print(f"🎯 Default rate: {y.mean():.3f}")
    
    return X, y

def final_data_cleaning(X, y):
    """
    Limpeza final robusta (baseada na sua função limpar_dados_final_para_modelo)
    """
    print("🧹 Aplicando limpeza final robusta...")
    
    X_clean = X.copy()
    y_clean = y.copy()
    
    # 1. Remover NaNs do target
    valid_target = ~y_clean.isnull()
    if not valid_target.all():
        print(f"   ⚠️ Removendo {(~valid_target).sum()} NaNs do target")
        X_clean = X_clean[valid_target]
        y_clean = y_clean[valid_target]
    
    # 2. Converter tudo para numérico
    print("   🔧 Convertendo para numérico...")
    for col in X_clean.columns:
        if X_clean[col].dtype == 'object':
            le = LabelEncoder()
            X_clean[col] = le.fit_transform(X_clean[col].astype(str))
    
    # 3. Tratar infinitos
    print("   🔧 Tratando infinitos...")
    for col in X_clean.select_dtypes(include=[np.number]).columns:
        if np.isinf(X_clean[col]).any():
            # Substituir infinitos por percentis
            valid_values = X_clean[col][np.isfinite(X_clean[col])]
            if len(valid_values) > 0:
                p1 = valid_values.quantile(0.01)
                p99 = valid_values.quantile(0.99)
                X_clean.loc[np.isposinf(X_clean[col]), col] = p99
                X_clean.loc[np.isneginf(X_clean[col]), col] = p1
            else:
                X_clean[col].replace([np.inf, -np.inf], 0, inplace=True)
    
    # 4. Tratar missing values
    print("   🔧 Tratando missing values...")
    for col in X_clean.columns:
        if X_clean[col].isnull().any():
            if X_clean[col].dtype in ['int64', 'float64']:
                fill_value = X_clean[col].median()
            else:
                fill_value = 0
            X_clean[col].fillna(fill_value, inplace=True)
    
    # 5. Garantir tipos corretos
    X_clean = X_clean.astype(np.float64)
    y_clean = y_clean.astype(int)
    
    # 6. Verificação final
    print("   ✅ Verificação final:")
    print(f"      X NaN: {X_clean.isnull().sum().sum()}")
    print(f"      X inf: {np.isinf(X_clean).sum().sum()}")
    print(f"      y NaN: {y_clean.isnull().sum()}")
    
    if X_clean.isnull().sum().sum() > 0 or np.isinf(X_clean).sum().sum() > 0:
        raise ValueError("❌ Ainda há problemas nos dados")
    
    print(f"   🎉 Dados limpos: {X_clean.shape}")
    return X_clean, y_clean

def train_logistic_regression(X, y, args):
    """
    Treina regressão logística (baseado no seu notebook)
    """
    print("🚂 Treinando Regressão Logística...")
    
    # Split para validação
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state, stratify=y
    )
    
    print(f"📊 Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Padronização (importante para LogisticRegression)
    print("📐 Padronizando features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Criar e treinar modelo
    print(f"🎯 Treinando com C={args.C}, penalty={args.penalty}...")
    model = LogisticRegression(
        C=args.C,
        penalty=args.penalty,
        random_state=args.random_state,
        max_iter=args.max_iter,
        solver='liblinear' if args.penalty == 'l1' else 'lbfgs',
        class_weight='balanced'  # Para dados desbalanceados
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Predições
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    
    # Métricas
    train_auc = roc_auc_score(y_train, y_train_proba)
    val_auc = roc_auc_score(y_val, y_val_proba)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    
    print(f"✅ RESULTADOS:")
    print(f"   Train AUC: {train_auc:.4f}")
    print(f"   Val AUC: {val_auc:.4f}")
    print(f"   Val Precision: {val_precision:.4f}")
    print(f"   Val Recall: {val_recall:.4f}")
    print(f"   Val F1: {val_f1:.4f}")
    print(f"   Overfitting: {train_auc - val_auc:.4f}")
    
    # Análise dos coeficientes (do seu notebook)
    print(f"\n🎯 TOP 10 FEATURES MAIS IMPORTANTES:")
    coef_df = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_[0],
        'odds_ratio': np.exp(model.coef_[0]),
        'abs_coef': np.abs(model.coef_[0])
    }).sort_values('abs_coef', ascending=False)
    
    feature_importance = coef_df.head(10).to_dict('records')
    
    for _, row in coef_df.head(10).iterrows():
        direction = "↗️ AUMENTA" if row['coefficient'] > 0 else "↘️ DIMINUI"
        print(f"   {row['feature']:<30} {direction} risco | OR: {row['odds_ratio']:6.3f}")
    
    # Score bands analysis (do seu notebook)
    print(f"\n📊 ANÁLISE SCORE BANDS:")
    df_analysis = pd.DataFrame({
        'score': y_val_proba,
        'default': y_val
    })
    
    try:
        # Criar quintis
        quintis = pd.qcut(df_analysis['score'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        score_bands = df_analysis.groupby(quintis)['default'].agg(['count', 'sum', 'mean']).round(4)
        score_bands.columns = ['Total', 'Defaults', 'Default_Rate']
        
        print(score_bands)
        
        # Verificar monotonicidade
        is_monotonic = score_bands['Default_Rate'].is_monotonic_increasing
        print(f"   Monotonicidade: {'✅ SIM' if is_monotonic else '❌ NÃO'}")
        
        if is_monotonic:
            print(f"   ✅ Modelo ordena corretamente o risco!")
        
    except Exception as e:
        print(f"   ⚠️ Erro na análise de score bands: {e}")
        score_bands = None
        is_monotonic = False
    
    # Métricas para salvar
    metrics = {
        'train_auc': float(train_auc),
        'val_auc': float(val_auc),
        'val_precision': float(val_precision),
        'val_recall': float(val_recall),
        'val_f1': float(val_f1),
        'gini': float(2 * val_auc - 1),
        'overfitting': float(train_auc - val_auc),
        'n_features': len(X.columns),
        'n_train_samples': len(X_train),
        'n_val_samples': len(X_val),
        'feature_importance': feature_importance,
        'is_monotonic': is_monotonic,
        'hyperparameters': {
            'C': args.C,
            'penalty': args.penalty,
            'max_iter': args.max_iter,
            'random_state': args.random_state
        },
        'training_date': datetime.now().isoformat()
    }
    
    return model, scaler, metrics

def save_model_and_artifacts(model, scaler, metrics, model_dir):
    """
    Salva modelo e artifacts
    """
    print(f"💾 Salvando modelo em: {model_dir}")
    
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Salvar modelo
    model_path = os.path.join(model_dir, 'model.joblib')
    joblib.dump(model, model_path)
    print(f"✅ Modelo salvo: {model_path}")
    
    # 2. Salvar scaler
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler salvo: {scaler_path}")
    
    # 3. Salvar métricas
    metrics_path = os.path.join(model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Métricas salvas: {metrics_path}")
    
    # 4. Criar script de inferência
    inference_script = '''
import joblib
import numpy as np
import pandas as pd
import os

def model_fn(model_dir):
    """Carrega modelo para inferência"""
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    return {'model': model, 'scaler': scaler}

def predict_fn(input_data, model_dict):
    """Faz predições"""
    model = model_dict['model']
    scaler = model_dict['scaler']
    
    # Aplicar scaling
    input_scaled = scaler.transform(input_data)
    
    # Predições
    probabilities = model.predict_proba(input_scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    
    return {
        'predictions': predictions.tolist(),
        'probabilities': probabilities.tolist(),
        'risk_scores': probabilities.tolist()
    }
'''
    
    inference_path = os.path.join(model_dir, 'inference.py')
    with open(inference_path, 'w') as f:
        f.write(inference_script)
    print(f"✅ Script de inferência salvo: {inference_path}")

def main():
    """
    Função principal - executa treinamento completo
    """
    print("🎯 CREDIT RISK TRAINING JOB")
    print("=" * 50)
    
    args = parse_args()
    print(f"⚙️ Hyperparameters: {vars(args)}")
    
    try:
        # 1. Carregar dados processados
        X, y = load_processed_data(args.train_dir)
        
        # 2. Limpeza final robusta
        X_clean, y_clean = final_data_cleaning(X, y)
        
        # 3. Treinar modelo
        model, scaler, metrics = train_logistic_regression(X_clean, y_clean, args)
        
        # 4. Salvar tudo
        save_model_and_artifacts(model, scaler, metrics, args.model_dir)
        
        print(f"\n🎉 TREINAMENTO CONCLUÍDO!")
        print(f"📊 AUC Final: {metrics['val_auc']:.4f}")
        print(f"📊 GINI: {metrics['gini']:.4f}")
        print(f"📦 Modelo salvo em: {args.model_dir}")
        
        # Status final
        if metrics['val_auc'] >= 0.75:
            print(f"🏆 EXCELENTE modelo!")
        elif metrics['val_auc'] >= 0.65:
            print(f"✅ BOM modelo!")
        else:
            print(f"⚠️ Modelo pode ser melhorado")
        
    except Exception as e:
        print(f"❌ Erro durante treinamento: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == '__main__':
    main()