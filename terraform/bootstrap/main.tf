###############################################################################
# Bootstrap — run ONCE, with LOCAL state.
#
# Creates the foundational plumbing that every other Terraform config depends
# on: the S3 bucket that will hold *their* remote state, plus a cost Budget so
# the sandbox account can't run up a surprise bill.
#
# This config keeps its own state LOCALLY (terraform.tfstate on disk) because
# the state bucket it creates does not exist yet — the classic bootstrap
# chicken-and-egg. Do NOT add a backend block here.
#
# Credentials come from the named AWS CLI profile (var.aws_profile); no access
# keys live in this code or in state.
###############################################################################

# Caller identity — used to scope the budget to this account and sanity-check
# we're pointed at the sandbox account, not a company one.
data "aws_caller_identity" "current" {}

#------------------------------------------------------------------------------
# S3 bucket for remote Terraform state
#------------------------------------------------------------------------------
resource "aws_s3_bucket" "tf_state" {
  bucket = var.state_bucket_name

  # Safety: never let `terraform destroy` nuke the bucket holding everyone's
  # state. To intentionally remove it you must flip this to false first.
  lifecycle {
    prevent_destroy = true
  }

  tags = {
    Project   = "rag-architecture-eval"
    Purpose   = "terraform-remote-state"
    ManagedBy = "terraform-bootstrap"
  }
}

# Keep historical versions of the state file (recovery from bad applies).
resource "aws_s3_bucket_versioning" "tf_state" {
  bucket = aws_s3_bucket.tf_state.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Encrypt state at rest (state can contain sensitive values).
resource "aws_s3_bucket_server_side_encryption_configuration" "tf_state" {
  bucket = aws_s3_bucket.tf_state.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block all public access — state must never be public.
resource "aws_s3_bucket_public_access_block" "tf_state" {
  bucket                  = aws_s3_bucket.tf_state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

#------------------------------------------------------------------------------
# Cost guardrail — monthly Budget with notifications
#------------------------------------------------------------------------------
resource "aws_budgets_budget" "monthly" {
  name         = "rag-arch-eval-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.budget_limit_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # Alert when actual spend crosses 80% of the limit.
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_email]
  }

  # Alert when AWS *forecasts* we'll exceed 100% by month end.
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_email]
  }
}
