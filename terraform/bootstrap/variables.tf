variable "aws_profile" {
  description = "Local AWS CLI profile to authenticate with (holds the IAM user 'dev' access key)."
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "state_bucket_name" {
  description = "Globally-unique S3 bucket name for remote Terraform state."
  type        = string
  default     = "rag-arch-eval-tfstate-875699230934"
}

variable "budget_limit_usd" {
  description = "Monthly cost budget limit in USD."
  type        = number
  default     = 20
}

variable "budget_email" {
  description = "Email address to receive budget alerts."
  type        = string
  # No default — supplied via terraform.tfvars (gitignored) or TF_VAR_budget_email.
}
