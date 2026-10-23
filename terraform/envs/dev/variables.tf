variable "aws_profile" {
  description = "Local AWS CLI profile (IAM user 'dev' access key)."
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "data_bucket_name" {
  description = "Globally-unique S3 bucket for the dataset (parquet) and retrieval indexes."
  type        = string
  default     = "rag-arch-eval-data-875699230934"
}
