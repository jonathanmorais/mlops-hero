# terraform/main.tf - SageMaker setup para Credit Risk

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.4"
    }
  }
}

# Variables
variable "project_name" {
  description = "Nome do projeto"
  type        = string
  default     = "credit-risk"
}

variable "environment" {
  description = "Ambiente (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "Região AWS"
  type        = string
  default     = "us-east-1"
}

variable "notebook_instance_type" {
  description = "Tipo de instância para notebooks"
  type        = string
  default     = "ml.t3.medium"  # $0.056/hora
}

variable "enable_notebook_instance" {
  description = "Criar SageMaker Notebook Instance (opcional)"
  type        = bool
  default     = false  # Use SageMaker Studio por padrão (mais barato)
}

variable "enable_vpc" {
  description = "Usar VPC específica"
  type        = bool
  default     = false
}

variable "subnet_id" {
  description = "Subnet ID (se enable_vpc = true)"
  type        = string
  default     = aws_subnets.private-us-east-1a.id  # Usar primeira subnet privada se não especificado
}

variable "allowed_ips" {
  description = "IPs permitidos para acessar recursos (CIDR)"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # ⚠️ Restringir em produção
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Provider
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
      CostCenter  = "ml-experimentation"
    }
  }
}

# Random suffix para recursos únicos
resource "random_id" "suffix" {
  byte_length = 4
}

# S3 Bucket para dados e modelos
resource "aws_s3_bucket" "ml_bucket" {
  bucket = "${var.project_name}-ml-${var.environment}-${random_id.suffix.hex}"
}

resource "aws_s3_bucket_versioning" "ml_bucket_versioning" {
  bucket = aws_s3_bucket.ml_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ml_bucket_encryption" {
  bucket = aws_s3_bucket.ml_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "ml_bucket_pab" {
  bucket = aws_s3_bucket.ml_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Criar estrutura de pastas no S3
resource "aws_s3_object" "folders" {
  for_each = toset([
    "data/raw/",
    "data/processed/",
    "data/features/",
    "models/",
    "artifacts/",
    "code/",
    "logs/"
  ])

  bucket = aws_s3_bucket.ml_bucket.id
  key    = each.value
  source = "/dev/null"  # Arquivo vazio para criar a pasta
}

# IAM Role para SageMaker
resource "aws_iam_role" "sagemaker_execution_role" {
  name = "${var.project_name}-sagemaker-execution-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

# Política customizada para o bucket S3
resource "aws_iam_role_policy" "sagemaker_s3_policy" {
  name = "${var.project_name}-sagemaker-s3-policy-${var.environment}"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          aws_s3_bucket.ml_bucket.arn,
          "${aws_s3_bucket.ml_bucket.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:*"
      }
    ]
  })
}

# Anexar políticas AWS gerenciadas
resource "aws_iam_role_policy_attachment" "sagemaker_execution_policy" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy_attachment" "sagemaker_cloudwatch_policy" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
}

# SageMaker Domain (preferível ao Notebook Instance)
resource "aws_sagemaker_domain" "ml_domain" {
  domain_name = "${var.project_name}-domain-${var.environment}"
  auth_mode   = "IAM"
  vpc_id      = aws_vpc.main.id
  subnet_ids  = [aws_subnet.private-us-east-1a.id, aws_subnet.private-us-east-1b.id]  # Subnets privadas

  default_user_settings {
    execution_role = aws_iam_role.sagemaker_execution_role.arn

    # Configurações de aplicações
    jupyter_server_app_settings {
      default_resource_spec {
        instance_type               = var.notebook_instance_type
        lifecycle_config_arn        = aws_sagemaker_studio_lifecycle_config.ml_lifecycle.arn
        sagemaker_image_version_arn = data.aws_sagemaker_prebuilt_ecr_image.datascience.image_uri
      }
    }

    kernel_gateway_app_settings {
      default_resource_spec {
        instance_type               = var.notebook_instance_type
        sagemaker_image_version_arn = data.aws_sagemaker_prebuilt_ecr_image.datascience.image_uri
      }
    }

    # Configurações de segurança
    sharing_settings {
      notebook_output_option = "Allowed"
      s3_output_path         = "s3://${aws_s3_bucket.ml_bucket.bucket}/shared-notebooks"
    }
  }

  default_space_settings {
    execution_role = aws_iam_role.sagemaker_execution_role.arn
  }

  retention_policy {
    home_efs_file_system = "Delete"  # Para economizar custos
  }
}

# User Profile para desenvolvimento
resource "aws_sagemaker_user_profile" "ml_user" {
  domain_id         = aws_sagemaker_domain.ml_domain.id
  user_profile_name = "${var.project_name}-user-${var.environment}"

  user_settings {
    execution_role = aws_iam_role.sagemaker_execution_role.arn

    jupyter_server_app_settings {
      default_resource_spec {
        instance_type = var.notebook_instance_type
      }
    }
  }
}

# Lifecycle Configuration para setup automático
resource "aws_sagemaker_studio_lifecycle_config" "ml_lifecycle" {
  studio_lifecycle_config_name     = "${var.project_name}-lifecycle-${var.environment}"
  studio_lifecycle_config_app_type = "JupyterServer"

  studio_lifecycle_config_content = base64encode(templatefile("${path.module}/lifecycle_config.sh", {
    bucket_name  = aws_s3_bucket.ml_bucket.bucket
    project_name = var.project_name
  }))
}

# Data source para imagem pré-construída
data "aws_sagemaker_prebuilt_ecr_image" "datascience" {
  repository_name = "datascience"
  image_tag       = "1.0"
}

# SageMaker Notebook Instance (opcional - menos recomendado)
resource "aws_sagemaker_notebook_instance" "ml_notebook" {
  count                   = var.enable_notebook_instance ? 1 : 0
  name                    = "${var.project_name}-notebook-${var.environment}"
  role_arn                = aws_iam_role.sagemaker_execution_role.arn
  instance_type           = var.notebook_instance_type
  platform_identifier     = "notebook-al2-v1"
  default_code_repository = aws_sagemaker_code_repository.ml_repo.code_repository_name

  # Configurações de rede (se VPC especificada)
  subnet_id              = var.enable_vpc ? var.subnet_id : null
  security_groups        = var.enable_vpc ? [aws_security_group.sagemaker_sg[0].id] : null
  direct_internet_access = var.enable_vpc ? "Disabled" : "Enabled"

  # Configurações de lifecycle
  lifecycle_config_name = aws_sagemaker_notebook_instance_lifecycle_configuration.ml_lifecycle_nb[0].name

  tags = {
    Name = "${var.project_name}-notebook-${var.environment}"
  }
}

# Lifecycle Configuration para Notebook Instance
resource "aws_sagemaker_notebook_instance_lifecycle_configuration" "ml_lifecycle_nb" {
  name  = "${var.project_name}-lifecycle-nb-${var.environment}"

  on_create = base64encode(templatefile("${path.module}/notebook_lifecycle.sh", {
    bucket_name  = aws_s3_bucket.ml_bucket.bucket
    project_name = var.project_name
  }))

  on_start = base64encode(templatefile("${path.module}/notebook_lifecycle.sh", {
    bucket_name  = aws_s3_bucket.ml_bucket.bucket
    project_name = var.project_name
  }))
}

# CodeCommit Repository para código
resource "aws_codecommit_repository" "ml_repo" {
  repository_name = "${var.project_name}-ml-repo-${var.environment}"
  description     = "Repository for ${var.project_name} ML code and notebooks"
}

# SageMaker Code Repository
resource "aws_sagemaker_code_repository" "ml_repo" {
  code_repository_name = "${var.project_name}-code-repo-${var.environment}"

  git_config {
    repository_url = aws_codecommit_repository.ml_repo.clone_url_http
  }
}

# Model Package Group para versionamento de modelos
resource "aws_sagemaker_model_package_group" "ml_model_group" {
  model_package_group_name        = "${var.project_name}-models-${var.environment}"
  model_package_group_description = "Model package group for ${var.project_name} credit risk models"

  tags = {
    Purpose = "credit-risk-modeling"
  }
}

# Security Group (apenas se VPC especificada)
resource "aws_security_group" "sagemaker_sg" {
  name        = "${var.project_name}-sagemaker-sg-${var.environment}"
  description = "Security group for SageMaker instances"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = var.allowed_ips
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-sagemaker-sg-${var.environment}"
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "ml_logs" {
  name              = "/aws/sagemaker/${var.project_name}-${var.environment}"
  retention_in_days = 30  # Para economizar custos

  tags = {
    Purpose = "ml-training-logs"
  }
}

# CloudWatch Alarm para monitorar custos
resource "aws_cloudwatch_metric_alarm" "cost_alarm" {
  alarm_name          = "${var.project_name}-cost-alarm-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = "86400"  # 24 horas
  statistic           = "Maximum"
  threshold           = "50"     # Alerta se passar de $50
  alarm_description   = "Alarm when charges exceed $50"
  alarm_actions       = []       # Adicionar SNS topic se necessário

  dimensions = {
    Currency = "USD"
  }

  tags = {
    Purpose = "cost-monitoring"
  }
}

# Outputs
output "s3_bucket_name" {
  description = "Nome do bucket S3 para dados e modelos"
  value       = aws_s3_bucket.ml_bucket.bucket
}

output "s3_bucket_arn" {
  description = "ARN do bucket S3"
  value       = aws_s3_bucket.ml_bucket.arn
}

output "sagemaker_execution_role_arn" {
  description = "ARN do role de execução do SageMaker"
  value       = aws_iam_role.sagemaker_execution_role.arn
}

output "sagemaker_domain_id" {
  description = "ID do SageMaker Domain"
  value       = aws_sagemaker_domain.ml_domain.id
}

output "sagemaker_domain_url" {
  description = "URL do SageMaker Studio"
  value       = "https://${aws_sagemaker_domain.ml_domain.domain_id}.studio.${data.aws_region.current.name}.sagemaker.aws/"
}

output "notebook_instance_name" {
  description = "Nome da instância do notebook (se criada)"
  value       = var.enable_notebook_instance ? aws_sagemaker_notebook_instance.ml_notebook[0].name : null
}

output "codecommit_repository_url" {
  description = "URL do repositório CodeCommit"
  value       = aws_codecommit_repository.ml_repo.clone_url_http
}

output "model_package_group_name" {
  description = "Nome do Model Package Group"
  value       = aws_sagemaker_model_package_group.ml_model_group.model_package_group_name
}

output "aws_region" {
  description = "Região AWS utilizada"
  value       = data.aws_region.current.name
}

output "account_id" {
  description = "ID da conta AWS"
  value       = data.aws_caller_identity.current.account_id
}

output "estimated_monthly_costs" {
  description = "Estimativa de custos mensais (USD)"
  value = {
    sagemaker_studio_minimal = "0-10 (notebooks sob demanda)"
    s3_storage_100gb         = "2-5"
    training_jobs_occasional = "5-20 (baseado na frequência)"
    cloudwatch_logs         = "1-3"
    total_dev_environment   = "8-38 USD/mês"
    notes = [
      "💡 Custos reais dependem do uso",
      "🚀 SageMaker Studio é gratuito quando não está em uso",
      "💰 Training jobs cobram apenas quando executando",
      "📊 S3 cobra por armazenamento e transferência"
    ]
  }
}

output "next_steps" {
  description = "Próximos passos após o deployment"
  value = [
    "1. 🌐 Acesse SageMaker Studio: ${aws_sagemaker_domain.ml_domain.domain_id}.studio.${data.aws_region.current.name}.sagemaker.aws/",
    "2. 📊 Faça upload dos dados para: s3://${aws_s3_bucket.ml_bucket.bucket}/data/raw/",
    "3. 💻 Clone seu código: git clone ${aws_codecommit_repository.ml_repo.clone_url_http}",
    "4. 🚂 Execute training jobs usando o SageMaker SDK",
    "5. 📦 Registre modelos no Model Package Group: ${aws_sagemaker_model_package_group.ml_model_group.model_package_group_name}",
    "6. 💰 Monitore custos no CloudWatch e AWS Cost Explorer"
  ]
}

output "useful_commands" {
  description = "Comandos úteis para desenvolvimento"
  value = [
    "# Upload de dados:",
    "aws s3 cp seus-dados.csv s3://${aws_s3_bucket.ml_bucket.bucket}/data/raw/",
    "",
    "# Listar training jobs:",
    "aws sagemaker list-training-jobs --name-contains ${var.project_name}",
    "",
    "# Verificar status de um job:",
    "aws sagemaker describe-training-job --training-job-name SEU-JOB-NAME",
    "",
    "# Baixar modelo treinado:",
    "aws s3 cp s3://${aws_s3_bucket.ml_bucket.bucket}/models/ ./models/ --recursive"
  ]
}