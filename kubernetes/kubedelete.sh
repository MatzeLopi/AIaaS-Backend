kubectl delete -f app-deployment.yaml
kubectl delete -f db-deployment.yaml
kubectl delete -f hpa.yaml
kubectl delete -f secrets.yaml
kubectl delete -f services.yaml
kubectl delete -f configmap.yaml
kubectl delete -f pv.yaml
kubectl delete -f pvc.yaml