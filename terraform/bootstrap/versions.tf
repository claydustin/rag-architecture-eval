terraform {
  required_version = ">= 1.10" # native S3 state locking (use_lockfile)
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}
