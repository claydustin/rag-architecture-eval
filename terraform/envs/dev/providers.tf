provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile

  default_tags {
    tags = {
      Project   = "rag-architecture-eval"
      Env       = "dev"
      ManagedBy = "terraform"
    }
  }
}
