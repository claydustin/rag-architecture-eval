output "data_bucket_name" {
  description = "S3 bucket for dataset parquet + retrieval indexes + eval results."
  value       = aws_s3_bucket.data.id
}

output "data_bucket_arn" {
  description = "ARN of the data bucket (reference from workload IAM policies)."
  value       = aws_s3_bucket.data.arn
}

output "account_id" {
  value = data.aws_caller_identity.current.account_id
}

output "sagemaker_domain_id" {
  description = "SageMaker Studio domain ID."
  value       = aws_sagemaker_domain.studio.id
}

output "sagemaker_domain_url" {
  description = "Studio domain landing URL (log in via the 'dev' user profile)."
  value       = aws_sagemaker_domain.studio.url
}

output "sagemaker_exec_role_arn" {
  description = "SageMaker execution role ARN."
  value       = aws_iam_role.sagemaker_exec.arn
}
