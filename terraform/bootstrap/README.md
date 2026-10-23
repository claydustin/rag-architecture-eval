# Bootstrap (run once, local state)

Creates the foundational plumbing the rest of the Terraform depends on:

- **S3 state bucket** (`rag-arch-eval-tfstate-875699230934`) — versioned,
  encrypted, public access blocked. Later configs (`terraform/envs/*`) point
  their remote backend at this bucket.
- **Monthly cost Budget** ($20) with email alerts at 80% actual and 100%
  forecasted.

This config stores its **own state locally** (`terraform.tfstate` on disk)
because the state bucket it creates doesn't exist yet. Do not add a backend
block here.

## Prerequisites

The `dev` AWS CLI profile must hold the IAM user `dev`'s access key:

```bash
aws configure --profile dev          # enter key id + secret, region us-east-1
aws sts get-caller-identity --profile dev   # expect account 875699230934
```

## Run

```bash
cd terraform/bootstrap
terraform init
terraform plan      # review: 6 resources to add
terraform apply
```

## Guardrails

- The state bucket has `prevent_destroy = true`. `terraform destroy` will refuse
  to delete it — intentional. To remove it you must edit the lifecycle block
  first (don't, unless you're tearing the whole project down).
- `terraform.tfvars` holds the budget email and is gitignored.
