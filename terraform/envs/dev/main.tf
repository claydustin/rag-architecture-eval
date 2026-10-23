###############################################################################
# dev environment — ongoing infrastructure for the RAG eval POC.
#
# State is REMOTE (S3 backend in versions.tf). This is where day-to-day
# init/plan/apply happens as the project grows.
#
# This file (Step 2b) provisions only the data bucket. IAM roles for the
# workloads (SageMaker, Fargate, Lambda) are added in the step that introduces
# each consumer (3b, 5, 10) so permissions stay scoped to real resource ARNs.
###############################################################################

data "aws_caller_identity" "current" {}

#------------------------------------------------------------------------------
# S3 data bucket — holds the dataset and retrieval artifacts.
#
# Prefix layout (prefixes are implicit in S3; documented here, not created):
#   data/      parquet corpus + queries (uploaded in Step 3a)
#   indexes/   BM25 / FAISS retrieval indexes (written by the dense phase)
#   results/   eval output (runs.csv + per-query CSV from Fargate jobs)
#------------------------------------------------------------------------------
resource "aws_s3_bucket" "data" {
  bucket = var.data_bucket_name

  tags = {
    Purpose = "rag-dataset-and-indexes"
  }
}

# Versioning on: cheap safety net for overwritten indexes/results.
resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
