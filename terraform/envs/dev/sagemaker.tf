###############################################################################
# SageMaker Studio playground (Step 3b)
#
# Interactive dev environment for building/testing the rag_eval retrieval and
# metric functions against the data bucket. First workload consumer, so the
# SageMaker execution role is defined here (option 2: roles live with their
# consumer).
#
# Cost guardrail: native idle shutdown is baked into the domain's default app
# settings (idle_settings), so JupyterLab apps auto-stop after inactivity even
# if someone forgets to stop them manually. The domain itself is free when
# idle; only running apps bill.
#
# Networking: uses the account's DEFAULT VPC + its subnets, looked up
# dynamically (not hardcoded) so this config stays portable to another account.
###############################################################################

#------------------------------------------------------------------------------
# Default VPC + subnets (dynamic lookup)
#------------------------------------------------------------------------------
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

#------------------------------------------------------------------------------
# SageMaker execution role — least privilege for the playground.
#   - read dataset + write indexes/results in the data bucket
#   - CloudWatch logs
#   - Bedrock invoke (embeddings/Claude/rerank) for when verification clears
#------------------------------------------------------------------------------
data "aws_iam_policy_document" "sagemaker_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker_exec" {
  name               = "rag-arch-eval-sagemaker-exec"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
}

data "aws_iam_policy_document" "sagemaker_data_access" {
  # List the data bucket.
  statement {
    sid       = "ListDataBucket"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn]
  }
  # Read/write objects in the data bucket.
  statement {
    sid = "ReadWriteDataObjects"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
    ]
    resources = ["${aws_s3_bucket.data.arn}/*"]
  }
  # Invoke Bedrock models. Must cover THREE resource shapes:
  #  - foundation-model/*            : direct model invokes (Titan embeddings)
  #  - inference-profile/*           : the cross-region profile ARN Claude needs
  #  - foundation-model/* in OTHER regions the profile routes to (us.* profiles
  #    fan out across us-east-1/us-east-2/us-west-2), hence the region wildcard
  #    on the foundation-model ARN below.
  statement {
    sid = "BedrockInvoke"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
    ]
    resources = [
      "arn:aws:bedrock:*::foundation-model/*",
      "arn:aws:bedrock:${var.aws_region}:${data.aws_caller_identity.current.account_id}:inference-profile/*",
    ]
  }
  # NOTE: all sagemaker:* actions (Processing-job self-inspection, Studio UI
  # list/describe/presigned-url, etc.) are granted by the AWS-managed
  # AmazonSageMakerFullAccess policy attached below — so no hand-rolled
  # sagemaker statements here. This inline policy keeps only what FullAccess
  # does NOT cover for us: our specific data bucket + Bedrock invoke.
}

resource "aws_iam_role_policy" "sagemaker_data_access" {
  name   = "data-and-bedrock-access"
  role   = aws_iam_role.sagemaker_exec.id
  policy = data.aws_iam_policy_document.sagemaker_data_access.json
}

# CloudWatch Logs for Studio apps.
resource "aws_iam_role_policy_attachment" "sagemaker_cloudwatch" {
  role       = aws_iam_role.sagemaker_exec.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
}

# Broad SageMaker access (AWS-managed). Covers Studio UI (list/describe apps +
# spaces, presigned domain URL), Processing-job APIs, and general SageMaker
# operations — so we don't accrete individual sagemaker:* actions by hand.
# Broader than least-privilege; acceptable for a sandbox POC. For the company
# handoff, consider replacing with a scoped policy.
resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_exec.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

#------------------------------------------------------------------------------
# Studio domain — native idle shutdown baked into default app settings.
#------------------------------------------------------------------------------
resource "aws_sagemaker_domain" "studio" {
  domain_name = "rag-arch-eval"
  auth_mode   = "IAM"
  vpc_id      = data.aws_vpc.default.id
  subnet_ids  = data.aws_subnets.default.ids

  default_user_settings {
    execution_role = aws_iam_role.sagemaker_exec.arn

    # JupyterLab app defaults, including NATIVE idle auto-shutdown.
    jupyter_lab_app_settings {
      default_resource_spec {
        instance_type = "ml.t3.medium"
      }

      app_lifecycle_management {
        idle_settings {
          # Auto-stop an idle JupyterLab app after 60 minutes.
          # AWS requires min_idle_timeout_in_minutes >= 60.
          idle_timeout_in_minutes      = 60
          lifecycle_management         = "ENABLED"
          max_idle_timeout_in_minutes  = 120
          min_idle_timeout_in_minutes  = 60
        }
      }
    }
  }

  # Same idle shutdown applied at the space level (spaces hold the running app).
  default_space_settings {
    execution_role = aws_iam_role.sagemaker_exec.arn

    jupyter_lab_app_settings {
      default_resource_spec {
        instance_type = "ml.t3.medium"
      }

      app_lifecycle_management {
        idle_settings {
          idle_timeout_in_minutes     = 60
          lifecycle_management        = "ENABLED"
          max_idle_timeout_in_minutes = 120
          min_idle_timeout_in_minutes = 60
        }
      }
    }
  }
}

#------------------------------------------------------------------------------
# User profile — how you log into Studio.
#------------------------------------------------------------------------------
resource "aws_sagemaker_user_profile" "dev" {
  domain_id         = aws_sagemaker_domain.studio.id
  user_profile_name = "dev"

  user_settings {
    execution_role = aws_iam_role.sagemaker_exec.arn
  }
}
