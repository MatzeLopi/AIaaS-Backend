kubectl apply -f pv.yaml
kubectl apply -f pvc.yaml
kubectl apply -f hpa.yaml
kubectl apply -f secrets.yaml
kubectl apply -f services.yaml
kubectl apply -f configmap.yaml
kubectl apply -f app-deployment.yaml
kubectl apply -f db-deployment.yaml
