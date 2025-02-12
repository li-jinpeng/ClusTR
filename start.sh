cluster_amount=3
gpu1=0
gpu2=1
for data in ETTh1
    do
        for model in PatchTST
            do
                nohup bash ./scripts/clustr_forecast/$data/$model.sh $gpu2 $cluster_amount > ./logs/$data/$cluster_amount.$model.log 2>&1 &
                # nohup bash ./scripts/clustr_baseline/$data/$model.sh $gpu1 > ./logs/$data/baseline.$model.log 2>&1 &
        done
done
