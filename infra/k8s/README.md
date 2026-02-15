# Kubernetes Deployment for SynApps v0.4.0

This directory contains Kubernetes manifests for deploying SynApps to a Kubernetes cluster. These instructions will guide you through the process of setting up SynApps in a production-ready Kubernetes environment.

## Prerequisites

- A Kubernetes cluster (e.g., GKE, EKS, AKS, or local kind/minikube)
- [kubectl](https://kubernetes.io/docs/tasks/tools/) configured to access your cluster
- Docker images pushed to a container registry (e.g., [GitHub Container Registry](https://ghcr.io), Docker Hub, or your private registry)
- Basic understanding of Kubernetes concepts (Deployments, Services, ConfigMaps, Secrets)

## Configuration

### 1. Prepare Secrets

SynApps requires API keys for OpenAI and Stability AI services. These must be stored securely as Kubernetes secrets:

1. Copy the secrets template and update with your values:
   ```bash
   cp secrets.yaml.template secrets.yaml
   ```

2. Edit `secrets.yaml` and replace the placeholder values with base64-encoded API keys:
   
   **Linux/macOS:**
   ```bash
   echo -n "your_openai_api_key" | base64
   echo -n "your_stability_api_key" | base64
   ```
   
   **Windows (PowerShell):**
   ```powershell
   [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("your_openai_api_key"))
   [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("your_stability_api_key"))
   ```

3. Apply the secrets to your cluster:
   ```bash
   kubectl apply -f secrets.yaml
   ```

### 2. Configure Environment Variables

Review and update the environment variables in `config.yaml` to match your deployment environment:

```bash
kubectl apply -f config.yaml
```

## Deployment

### 1. Deploy the Database

SynApps uses a PostgreSQL database in production. If you don't have an existing database, you can deploy one in your cluster:

```bash
# Example for deploying PostgreSQL (if needed)
kubectl apply -f https://raw.githubusercontent.com/kubernetes/examples/master/staging/postgresql/postgres-deployment.yaml
kubectl apply -f https://raw.githubusercontent.com/kubernetes/examples/master/staging/postgresql/postgres-service.yaml
```

### 2. Deploy the Orchestrator

The orchestrator is the core backend service of SynApps:

```bash
kubectl apply -f orchestrator.yaml
```

### 3. Set Up Autoscaling

Configure horizontal pod autoscaling to handle varying loads:

```bash
kubectl apply -f autoscaling.yaml
```

### 4. Deploy the Frontend (Optional)

If you're deploying the frontend in the same cluster:

```bash
# Example - adjust based on your frontend configuration
kubectl apply -f frontend.yaml
```

## Verifying the Deployment

### 1. Check Resource Status

Verify that all resources are running correctly:

```bash
# Check deployments
kubectl get deployments

# Check pods and their status
kubectl get pods

# Check services and their endpoints
kubectl get services

# Check horizontal pod autoscalers
kubectl get hpa
```

### 2. Verify Logs

Check the logs to ensure the application is running without errors:

```bash
# Get logs from the orchestrator pods
kubectl logs -l app=synapps-orchestrator
```

### 3. Test the API

Test that the API is accessible and responding correctly:

```bash
# Port-forward to access the API locally
kubectl port-forward svc/synapps-orchestrator-service 8000:80

# In another terminal, test the API
curl http://localhost:8000/health
```

## Accessing the API

### Using Ingress

The API is exposed through an Ingress controller. Configure your DNS to point to the ingress controller's IP address.

1. Get the IP address of your ingress controller:

   ```bash
   kubectl get ingress synapps-orchestrator-ingress
   ```

2. Update your DNS settings to point `api.yourdomain.com` to this IP address.

3. Access the API at `https://api.yourdomain.com`.

### Using Port Forwarding (Development)

For local development or testing, you can use port forwarding:

```bash
kubectl port-forward svc/synapps-orchestrator-service 8000:80
```

Then access the API at `http://localhost:8000`.

## Scaling and Management

### Horizontal Pod Autoscaling

The HorizontalPodAutoscaler automatically scales the number of pods based on CPU and memory utilization. You can adjust the scaling parameters in `autoscaling.yaml`.

To view current autoscaling metrics:

```bash
kubectl describe hpa synapps-orchestrator-hpa
```

### Resource Management

Monitor resource usage to ensure your cluster has sufficient capacity:

```bash
# View resource usage across nodes
kubectl top nodes

# View resource usage across pods
kubectl top pods
```

### Updating the Deployment

To update to a new version of SynApps:

```bash
# Update the image in the deployment
kubectl set image deployment/synapps-orchestrator synapps-orchestrator=ghcr.io/nxtg-ai/synapps-orchestrator:new-tag

# Monitor the rollout
kubectl rollout status deployment/synapps-orchestrator
```

## Troubleshooting

### Common Issues

#### Pods Not Starting

If pods are stuck in `Pending` or `ContainerCreating` state:

```bash
# Get detailed pod information
kubectl describe pod <pod-name>

# Check events related to the pod
kubectl get events --sort-by=.metadata.creationTimestamp | grep <pod-name>
```

#### API Not Accessible

If you can't access the API:

1. Check if the service is running:
   ```bash
   kubectl get svc synapps-orchestrator-service
   ```

2. Check if the pods are running:
   ```bash
   kubectl get pods -l app=synapps-orchestrator
   ```

3. Check the logs for errors:
   ```bash
   kubectl logs -l app=synapps-orchestrator
   ```

#### Database Connection Issues

If the application can't connect to the database:

```bash
# Check database service
kubectl get svc postgres

# Check database pod
kubectl get pods -l app=postgres

# Check database logs
kubectl logs -l app=postgres
```

### Diagnostic Commands

```bash
# Check all events in the cluster
kubectl get events --sort-by=.metadata.creationTimestamp

# Check pod logs
kubectl logs -l app=synapps-orchestrator --tail=100

# Check pod details
kubectl describe pod -l app=synapps-orchestrator
```

## Cleanup

To remove all deployed resources:

```bash
# Remove autoscaling
kubectl delete -f autoscaling.yaml

# Remove orchestrator deployment and service
kubectl delete -f orchestrator.yaml

# Remove secrets
kubectl delete -f secrets.yaml

# Remove config
kubectl delete -f config.yaml
```

## Security Considerations

### Secret Management

For production deployments, consider using a more secure secret management solution:

- [HashiCorp Vault](https://www.vaultproject.io/)
- [AWS Secrets Manager](https://aws.amazon.com/secrets-manager/) (for EKS)
- [Azure Key Vault](https://azure.microsoft.com/en-us/services/key-vault/) (for AKS)
- [Google Secret Manager](https://cloud.google.com/secret-manager) (for GKE)

### Network Security

- Use network policies to restrict pod-to-pod communication
- Enable TLS for all ingress traffic
- Consider using a service mesh like Istio for advanced traffic management

## Monitoring and Logging

For production deployments, consider setting up:

- [Prometheus](https://prometheus.io/) for metrics collection
- [Grafana](https://grafana.com/) for metrics visualization
- [ELK Stack](https://www.elastic.co/elastic-stack) or [Loki](https://grafana.com/oss/loki/) for log aggregation
