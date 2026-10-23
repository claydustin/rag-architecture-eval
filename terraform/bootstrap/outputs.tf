output "state_bucket_name" {
  description = "Name of the S3 bucket holding remote Terraform state. Reference this as the backend bucket in terraform/envs/*."
  value       = aws_s3_bucket.tf_state.id
}

output "account_id" {
  description = "AWS account ID these resources were created in (sanity-check: sandbox, not a company account)."
  value       = data.aws_caller_identity.current.account_id
}

output "budget_name" {
  description = "Name of the monthly cost budget."
  value       = aws_budgets_budget.monthly.name
}
