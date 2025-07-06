#!/usr/bin/env python3
"""
src/pipeline.py - Orquestrador principal baseado no seu notebook

Este arquivo orquestra todo o processo:
1. 📊 Data Processing (local) - todo seu código de limpeza e feature engineering
2. 🚂 Training (SageMaker) - executa train.py no SageMaker
3. 📈 Evaluation (local) - suas análises e visualizações

Vantagem: Usa SageMaker só para treinar (barato!)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sagemaker
import boto3
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput
import joblib
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import os

class CreditRiskPipeline:
    def __init__(self, bucket_name=None):
        """Inicializa pipeline de Credit Risk"""
        self.sess = sagemaker.Session()
        self.bucket = bucket_name or self.sess.default_bucket()
        
        try:
            self.role = sagemaker.get_execution_role()
        except:
            # Se não estiver no SageMaker, usar role específico
            self.role = f"arn:aws:iam::{boto3.client('sts').get_caller_identity()['Account']}:role/SageMakerExecutionRole"
        
        self.region = self.sess.boto_region_name
        
        print(f"🚀 Credit Risk Pipeline inicializada")
        print(f"📦 Bucket: {self.bucket}")
        print(f"🌍 Região: {self.region}")
        print(f"🔑 Role: {self.role}")

    def executar_pipeline_completo(self, data_path):
        """
        Executa pipeline completo baseado no seu notebook
        """
        print("🎯 EXECUTANDO PIPELINE COMPLETO DE CREDIT RISK")
        print("=" * 60)
        
        try:
            # 1. CARREGAR E LIMPAR DADOS (do seu notebook)
            print("\n1️⃣ CARREGANDO E LIMPANDO DADOS...")
            df_clean = self.carregar_e_limpar_dados_inicial(data_path)
            
            # 2. CORREÇÃO MONTHLY INCOME (do seu notebook)
            print("\n2️⃣ CORRIGINDO MONTHLY INCOME...")
            df_income_corrigido = self.corrigir_monthly_income_completo(df_clean)
            
            # 3. FEATURE ENGINEERING (do seu notebook)
            print("\n3️⃣ FEATURE ENGINEERING COMPLETO...")
            df_features = self.feature_engineering_completo(df_income_corrigido)
            
            # 4. ANÁLISE BIVARIADA (do seu notebook)
            print("\n4️⃣ ANÁLISE BIVARIADA...")
            correlacoes = self.analise_bivariada_rapida(df_features)
            
            # 5. PREPARAR DADOS PARA TREINAMENTO
            print("\n5️⃣ PREPARANDO DADOS PARA SAGEMAKER...")
            train_s3_uri, test_s3_uri = self.preparar_dados_treinamento(df_features)
            
            # 6. TREINAR MODELO NO SAGEMAKER
            print("\n6️⃣ TREINANDO MODELO NO SAGEMAKER...")
            estimator = self.treinar_modelo_sagemaker(train_s3_uri)
            
            # 7. AVALIAÇÃO COMPLETA (do seu notebook)
            print("\n7️⃣ AVALIAÇÃO COMPLETA DO MODELO...")
            metricas_finais = self.avaliar_modelo_completo(estimator, test_s3_uri)
            
            print("\n🎉 PIPELINE CONCLUÍDO COM SUCESSO!")
            return {
                'estimator': estimator,
                'metricas': metricas_finais,
                'correlacoes': correlacoes,
                'train_data_uri': train_s3_uri,
                'test_data_uri': test_s3_uri
            }
            
        except Exception as e:
            print(f"\n❌ ERRO NO PIPELINE: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def carregar_e_limpar_dados_inicial(self, data_path):
        """
        Carregamento e limpeza inicial (do seu notebook)
        """
        print("📊 Carregando dados...")
        
        # Carregar dados (do seu notebook)
        data_df = pd.read_csv(data_path)
        print(f"   Dados originais: {data_df.shape}")
        
        # Remover linhas com missing (do seu notebook)
        data_df = data_df.dropna(axis=0, how='any')
        print(f"   Após remover NaN: {data_df.shape}")
        
        # Remover colunas desnecessárias (do seu notebook)
        if 'Unnamed: 0' in data_df.columns:
            data_df = data_df.drop(columns=['Unnamed: 0'])
            print(f"   Removido 'Unnamed: 0'")
        
        print(f"✅ Dados limpos: {data_df.shape}")
        print(f"📊 Distribuição target: {data_df['SeriousDlqin2yrs'].value_counts().to_dict()}")
        print(f"🎯 Taxa de default: {data_df['SeriousDlqin2yrs'].mean():.3f}")
        
        return data_df

    def corrigir_monthly_income_completo(self, df):
        """
        Correção completa de MonthlyIncome (baseado na sua função)
        """
        print("🔧 Corrigindo MonthlyIncome...")
        
        df_corrigido = df.copy()
        income_col = 'MonthlyIncome'
        target_col = 'SeriousDlqin2yrs'
        
        # Diagnóstico inicial
        valores_negativos = (df_corrigido[income_col] < 0).sum()
        correlacao_original = df_corrigido[income_col].corr(df_corrigido[target_col])
        
        print(f"   📊 Diagnóstico inicial:")
        print(f"      Valores negativos: {valores_negativos}")
        print(f"      Correlação original: {correlacao_original:.4f}")
        
        if valores_negativos > 0:
            print(f"   🔄 Corrigindo {valores_negativos} valores negativos...")
            
            # Substituir negativos por NaN
            df_corrigido.loc[df_corrigido[income_col] < 0, income_col] = np.nan
            
            # Imputação inteligente por faixa etária (do seu código)
            if 'age' in df_corrigido.columns:
                print(f"      Usando mediana por faixa etária...")
                
                df_corrigido['faixa_etaria_temp'] = pd.cut(
                    df_corrigido['age'], 
                    bins=[0, 25, 35, 45, 55, 100], 
                    labels=['18-25', '26-35', '36-45', '46-55', '55+']
                )
                
                # Calcular medianas por faixa
                for faixa in df_corrigido['faixa_etaria_temp'].unique():
                    if pd.notna(faixa):
                        mask_faixa = df_corrigido['faixa_etaria_temp'] == faixa
                        mask_valid = df_corrigido[income_col] > 0
                        
                        if (mask_faixa & mask_valid).sum() > 0:
                            mediana_faixa = df_corrigido.loc[mask_faixa & mask_valid, income_col].median()
                            mask_missing = df_corrigido[income_col].isnull()
                            
                            count_imputados = (mask_faixa & mask_missing).sum()
                            if count_imputados > 0:
                                df_corrigido.loc[mask_faixa & mask_missing, income_col] = mediana_faixa
                                print(f"         {faixa}: {count_imputados} valores imputados com ${mediana_faixa:,.0f}")
                
                df_corrigido.drop('faixa_etaria_temp', axis=1, inplace=True)
            
            # Preencher restantes com mediana geral
            missing_restantes = df_corrigido[income_col].isnull().sum()
            if missing_restantes > 0:
                mediana_geral = df_corrigido[income_col].median()
                df_corrigido[income_col].fillna(mediana_geral, inplace=True)
                print(f"      {missing_restantes} restantes imputados com mediana geral: ${mediana_geral:,.0f}")
        
        # Verificar correlação após correção
        correlacao_corrigida = df_corrigido[income_col].corr(df_corrigido[target_col])
        
        print(f"   ✅ Resultado:")
        print(f"      Correlação após correção: {correlacao_corrigida:.4f}")
        print(f"      Melhoria: {abs(correlacao_corrigida) - abs(correlacao_original):.4f}")
        
        return df_corrigido

    def feature_engineering_completo(self, df):
        """
        Feature engineering completo (baseado na sua função criar_features_give_me_credit_completo)
        """
        print("⚙️ Executando feature engineering completo...")
        
        df_features = df.copy()
        features_criadas = []
        
        # 1. FEATURES DE IDADE (do seu código)
        if 'age' in df.columns:
            print("   👤 Criando features de idade...")
            
            # Faixas etárias
            bins_idade = [0, 25, 35, 50, 65, 100]
            labels_idade = [0, 1, 2, 3, 4]  # Já encoded
            df_features['age_group_encoded'] = pd.cut(df['age'], bins=bins_idade, 
                                                     labels=labels_idade, include_lowest=True)
            
            # Flags de risco
            df_features['age_very_young'] = (df['age'] <= 25).astype(int)
            df_features['age_prime'] = ((df['age'] >= 30) & (df['age'] <= 50)).astype(int)
            df_features['age_senior'] = (df['age'] >= 60).astype(int)
            df_features['age_squared'] = df['age'] ** 2
            
            features_criadas.extend(['age_group_encoded', 'age_very_young', 'age_prime', 'age_senior', 'age_squared'])
        
        # 2. FEATURES DE RENDA (do seu código)
        if 'MonthlyIncome' in df.columns:
            print("   💰 Criando features de renda...")
            
            median_income = df['MonthlyIncome'].median()
            mean_income = df['MonthlyIncome'].mean()
            
            df_features['low_income'] = (df['MonthlyIncome'] <= median_income * 0.5).astype(int)
            df_features['high_income'] = (df['MonthlyIncome'] >= mean_income * 1.5).astype(int)
            df_features['very_high_income'] = (df['MonthlyIncome'] >= df['MonthlyIncome'].quantile(0.9)).astype(int)
            df_features['income_relative'] = (df['MonthlyIncome'] - mean_income) / df['MonthlyIncome'].std()
            
            features_criadas.extend(['low_income', 'high_income', 'very_high_income', 'income_relative'])
        
        # 3. FEATURES DE COMPORTAMENTO (do seu código)
        comportamento_cols = [
            'NumberOfTime30-59DaysPastDueNotWorse',
            'NumberOfTime60-89DaysPastDueNotWorse', 
            'NumberOfTimes90DaysLate'
        ]
        
        if all(col in df.columns for col in comportamento_cols):
            print("   📈 Criando features de comportamento...")
            
            # Total de atrasos
            df_features['total_delinquencies'] = df[comportamento_cols].sum(axis=1)
            df_features['has_any_delinquency'] = (df_features['total_delinquencies'] > 0).astype(int)
            
            # Tipos de atraso
            df_features['total_delinquency_types'] = (
                (df[comportamento_cols[0]] > 0).astype(int) +
                (df[comportamento_cols[1]] > 0).astype(int) +
                (df[comportamento_cols[2]] > 0).astype(int)
            )
            df_features['severe_delinquency_pattern'] = (df_features['total_delinquency_types'] >= 2).astype(int)
            
            features_criadas.extend(['total_delinquencies', 'has_any_delinquency', 
                                   'total_delinquency_types', 'severe_delinquency_pattern'])
        
        # 4. FEATURES DE UTILIZAÇÃO DE CRÉDITO (do seu código)
        if 'RevolvingUtilizationOfUnsecuredLines' in df.columns:
            print("   💳 Criando features de utilização de crédito...")
            
            df_features['credit_util_low'] = (df['RevolvingUtilizationOfUnsecuredLines'] <= 0.3).astype(int)
            df_features['credit_util_high'] = (df['RevolvingUtilizationOfUnsecuredLines'] >= 0.7).astype(int)
            df_features['credit_util_maxed'] = (df['RevolvingUtilizationOfUnsecuredLines'] >= 0.95).astype(int)
            df_features['credit_util_squared'] = df['RevolvingUtilizationOfUnsecuredLines'] ** 2
            
            features_criadas.extend(['credit_util_low', 'credit_util_high', 'credit_util_maxed', 'credit_util_squared'])
        
        # 5. FEATURES DE DEBT RATIO (do seu código)
        if 'DebtRatio' in df.columns:
            print("   💸 Criando features de debt ratio...")
            
            df_features['debt_ratio_safe'] = (df['DebtRatio'] <= 0.3).astype(int)
            df_features['debt_ratio_risky'] = (df['DebtRatio'] >= 0.5).astype(int)
            
            features_criadas.extend(['debt_ratio_safe', 'debt_ratio_risky'])
        
        # 6. FEATURES DE DEPENDENTES (do seu código)
        if 'NumberOfDependents' in df.columns:
            print("   👨‍👩‍👧‍👦 Criando features de dependentes...")
            
            df_features['has_dependents'] = (df['NumberOfDependents'] > 0).astype(int)
            df_features['many_dependents'] = (df['NumberOfDependents'] >= 3).astype(int)
            
            features_criadas.extend(['has_dependents', 'many_dependents'])
        
        # 7. FEATURES DE INTERAÇÃO (do seu código)
        if 'age' in df.columns and 'MonthlyIncome' in df.columns:
            print("   🔗 Criando features de interação...")
            
            df_features['income_per_age'] = df['MonthlyIncome'] / (df['age'] + 1)
            features_criadas.append('income_per_age')
        
        if 'NumberOfDependents' in df.columns and 'MonthlyIncome' in df.columns:
            df_features['income_per_dependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
            features_criadas.append('income_per_dependent')
        
        # 8. FEATURES DE RISCO COMBINADO (do seu código)
        print("   ⚠️ Criando features de risco combinado...")
        
        # Score de risco comportamental
        if 'has_any_delinquency' in df_features.columns:
            df_features['behavioral_risk_score'] = (
                df_features.get('has_any_delinquency', 0) * 3 +
                df_features.get('severe_delinquency_pattern', 0) * 2 +
                df_features.get('total_delinquency_types', 0)
            )
            features_criadas.append('behavioral_risk_score')
        
        # Score de risco financeiro
        financial_components = []
        if 'low_income' in df_features.columns:
            financial_components.append('low_income')
        if 'debt_ratio_risky' in df_features.columns:
            financial_components.append('debt_ratio_risky')
        if 'credit_util_high' in df_features.columns:
            financial_components.append('credit_util_high')
        
        if financial_components:
            df_features['financial_risk_score'] = df_features[financial_components].sum(axis=1)
            features_criadas.append('financial_risk_score')
        
        print(f"   ✅ {len(features_criadas)} features criadas")
        print(f"   📊 Shape final: {df_features.shape}")
        
        return df_features

    def analise_bivariada_rapida(self, df):
        """
        Análise bivariada rápida (baseada no seu notebook)
        """
        print("🔍 Executando análise bivariada...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'SeriousDlqin2yrs' in numeric_cols:
            numeric_cols.remove('SeriousDlqin2yrs')
        
        # Calcular correlações
        corr_with_target = df[numeric_cols + ['SeriousDlqin2yrs']].corr()['SeriousDlqin2yrs'].sort_values(ascending=False)
        
        print("🎯 Top 10 correlações com target:")
        for var, corr in corr_with_target.head(10).items():
            if var != 'SeriousDlqin2yrs':
                direction = "↗️" if corr > 0 else "↘️"
                strength = "FORTE" if abs(corr) > 0.3 else "MODERADA" if abs(corr) > 0.1 else "FRACA"
                print(f"   {var:<35} {direction} {corr:7.4f} ({strength})")
        
        return corr_with_target.to_dict()

    def preparar_dados_treinamento(self, df):
        """
        Prepara dados para SageMaker (divisão + limpeza + upload)
        """
        print("🔧 Preparando dados para SageMaker...")
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        
        # Separar features e target
        X = df.drop('SeriousDlqin2yrs', axis=1)
        y = df['SeriousDlqin2yrs']
        
        print(f"   📊 Features: {X.shape}")
        print(f"   🎯 Target: {y.shape}")
        
        # Limpeza adicional para garantir compatibilidade
        print("   🧹 Limpeza para SageMaker...")
        
        # Tratar colunas categóricas restantes
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                print(f"      Convertendo {col} para numérico...")
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Tratar missing values básicos
        X = X.fillna(X.median())
        
        # Tratar infinitos
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        
        # Garantir tipos
        X = X.astype(np.float64)
        y = y.astype(int)
        
        # Split estratificado
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   📊 Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"   🎯 Train default rate: {y_train.mean():.3f}")
        print(f"   🎯 Test default rate: {y_test.mean():.3f}")
        
        # Criar datasets completos
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        # Salvar localmente
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_file = f'train_data_{timestamp}.csv'
        test_file = f'test_data_{timestamp}.csv'
        
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        
        print(f"   💾 Dados salvos: {train_file}, {test_file}")
        
        # Upload para S3
        train_s3_uri = self.sess.upload_data(
            path=train_file, bucket=self.bucket, key_prefix='credit-risk/data'
        )
        test_s3_uri = self.sess.upload_data(
            path=test_file, bucket=self.bucket, key_prefix='credit-risk/data'
        )
        
        print(f"   📤 Train S3: {train_s3_uri}")
        print(f"   📤 Test S3: {test_s3_uri}")
        
        # Limpar arquivos locais
        os.remove(train_file)
        os.remove(test_file)
        
        return train_s3_uri, test_s3_uri

    def treinar_modelo_sagemaker(self, train_s3_uri):
        """
        Treina modelo no SageMaker (só a parte cara!)
        """
        print("🚂 Iniciando treinamento no SageMaker...")
        
        # Criar estimator
        estimator = SKLearn(
            entry_point='train.py',
            source_dir='src',  # Pasta onde está o train.py
            framework_version='0.23-1',
            py_version='py3',
            instance_type='ml.m5.large',  # $0.23/hora
            instance_count=1,
            role=self.role,
            sagemaker_session=self.sess,
            
            # Hyperparameters (do seu notebook)
            hyperparameters={
                'C': 0.1,  # Regularização da LogisticRegression
                'penalty': 'l2',
                'max_iter': 1000,
                'random_state': 42
            }
        )
        
        # Configurar dados de entrada
        training_input = TrainingInput(
            s3_data=train_s3_uri,
            content_type='text/csv'
        )
        
        # Nome do job
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"credit-risk-{timestamp}"
        
        print(f"🚀 Executando training job: {job_name}")
        print(f"💰 Custo estimado: ~$0.50-1.00 (2h × $0.23/hora)")
        print(f"⏰ Tempo estimado: 5-15 minutos")
        
        # Iniciar treinamento
        estimator.fit(
            inputs={'training': training_input},
            job_name=job_name,
            wait=True  # Aguardar conclusão
        )
        
        print(f"✅ Treinamento concluído!")
        print(f"📦 Modelo salvo em: {estimator.model_data}")
        
        return estimator

    def avaliar_modelo_completo(self, estimator, test_s3_uri):
        """
        Avaliação completa do modelo (baseada no seu notebook)
        """
        print("📊 Avaliando modelo...")
        
        # Baixar dados de teste
        test_data = pd.read_csv(test_s3_uri)
        X_test = test_data.drop('SeriousDlqin2yrs', axis=1)
        y_test = test_data['SeriousDlqin2yrs']
        
        print(f"   📊 Dados de teste: {X_test.shape}")
        
        # Criar predictor temporário para avaliação
        predictor = estimator.deploy(
            initial_instance_count=1,
            instance_type='ml.t2.medium',  # Mais barato para inferência
            endpoint_name=f"temp-eval-{int(datetime.now().timestamp())}"
        )
        
        try:
            print("   🔮 Fazendo predições...")
            
            # Fazer predições em batches (para evitar timeout)
            batch_size = 1000
            all_predictions = []
            
            for i in range(0, len(X_test), batch_size):
                batch_X = X_test.iloc[i:i+batch_size]
                batch_pred = predictor.predict(batch_X.values)
                all_predictions.extend(batch_pred)
            
            # Processar predições
            if isinstance(all_predictions[0], dict):
                y_proba = [pred['probabilities'][0] if isinstance(pred['probabilities'], list) else pred['probabilities'] for pred in all_predictions]
                y_pred = [pred['predictions'][0] if isinstance(pred['predictions'], list) else pred['predictions'] for pred in all_predictions]
            else:
                y_proba = all_predictions
                y_pred = (np.array(y_proba) >= 0.5).astype(int)
            
            # Calcular métricas (do seu notebook)
            from sklearn.metrics import roc_auc_score, classification_report, roc_curve
            
            auc = roc_auc_score(y_test, y_proba)
            gini = 2 * auc - 1
            
            print(f"\n📈 RESULTADOS FINAIS:")
            print(f"   AUC: {auc:.4f}")
            print(f"   GINI: {gini:.4f}")
            
            # Classificar performance
            if auc >= 0.8:
                status = "🏆 EXCELENTE"
            elif auc >= 0.7:
                status = "✅ BOM"
            elif auc >= 0.6:
                status = "⚠️ ACEITÁVEL"
            else:
                status = "❌ PRECISA MELHORAR"
            
            print(f"   Status: {status}")
            
            # Classification report
            print(f"\n📋 Classification Report:")
            print(classification_report(y_test, y_pred))
            
            # Análise por score bands (do seu notebook)
            print(f"\n📊 Análise por Score Bands:")
            self.analisar_score_bands(y_test, y_proba)
            
            # Curva ROC rápida
            self.plotar_curva_roc_simples(y_test, y_proba, auc)
            
            metricas_finais = {
                'auc': auc,
                'gini': gini,
                'status': status,
                'n_test_samples': len(y_test),
                'test_default_rate': y_test.mean()
            }
            
            return metricas_finais
            
        finally:
            # IMPORTANTE: Deletar endpoint para não cobrar
            print("🗑️ Deletando endpoint temporário...")
            predictor.delete_endpoint()
            print("   ✅ Endpoint deletado (economizando custos)")

    def analisar_score_bands(self, y_true, y_proba):
        """
        Análise por score bands (do seu notebook)
        """
        df_analysis = pd.DataFrame({
            'score': y_proba,
            'default': y_true
        })
        
        try:
            # Criar quintis
            quintis = pd.qcut(df_analysis['score'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            score_bands = df_analysis.groupby(quintis)['default'].agg(['count', 'sum', 'mean']).round(4)
            score_bands.columns = ['Total', 'Defaults', 'Default_Rate']
            
            print(score_bands)
            
            # Verificar monotonicidade
            is_monotonic = score_bands['Default_Rate'].is_monotonic_increasing
            print(f"\n   Monotonicidade: {'✅ SIM' if is_monotonic else '❌ NÃO'}")
            
            if is_monotonic:
                print(f"   ✅ Modelo ordena corretamente o risco!")
            else:
                print(f"   ⚠️ Modelo não ordena perfeitamente o risco")
                
        except Exception as e:
            print(f"   ⚠️ Erro na análise: {e}")

    def plotar_curva_roc_simples(self, y_true, y_proba, auc):
        """
        Plot simples da curva ROC
        """
        from sklearn.metrics import roc_curve
        
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {auc:.3f})')
            plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random')
            plt.xlabel('Taxa de Falsos Positivos')
            plt.ylabel('Taxa de Verdadeiros Positivos')
            plt.title('Curva ROC - Credit Risk Model')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
        except Exception as e:
            print(f"   ⚠️ Erro ao plotar ROC: {e}")

def executar_pipeline_credit_risk(data_path):
    """
    Função principal para executar a pipeline completa
    """
    print("🎯 INICIANDO PIPELINE CREDIT RISK COMPLETA")
    print("=" * 60)
    
    # Inicializar pipeline
    pipeline = CreditRiskPipeline()
    
    # Executar pipeline completa
    resultados = pipeline.executar_pipeline_completo(data_path)
    
    print(f"\n🎉 PIPELINE CONCLUÍDA COM SUCESSO!")
    print(f"📊 AUC Final: {resultados['metricas']['auc']:.4f}")
    print(f"📊 Status: {resultados['metricas']['status']}")
    
    return resultados

# Para teste rápido
if __name__ == "__main__":
    # Teste da pipeline
    resultados = executar_pipeline_credit_risk("data/cs-training.csv")
    print("✅ Teste concluído!")