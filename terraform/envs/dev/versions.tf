terraform {
  required_version = ">= 1.10"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Remote state in the bucket created by terraform/bootstrap.
  # Native S3 state locking (use_lockfile) — no DynamoDB table needed (TF >=1.10).
  backend "s3" {
    bucket       = "rag-arch-eval-tfstate-875699230934"
    key          = "envs/dev/terraform.tfstate"
    region       = "us-east-1"
    profile      = "dev"
    encrypt      = true
    use_lockfile = true
  }
}
