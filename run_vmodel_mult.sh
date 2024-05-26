#!/bin/bash


vmodel --num-timesteps 3100 --verbose --progress --filter-occluded --num-agents 100 --delta-time 0.02 --num-preds 1 --pred-time 1500 --vision-pred 300 --flee-strength 50 --col-style 0 --num-runs 4 --pred-angle 90 --outfolder ~/vmodel_output/vids/side_noCol/

vmodel --num-timesteps 3100 --verbose --progress --filter-occluded --num-agents 100 --delta-time 0.02 --num-preds 1 --pred-time 1500 --vision-pred 300 --flee-strength 50 --col-style 0 --num-runs 4 --pred-angle 0 --outfolder ~/vmodel_output/vids/front_noCol/

vmodel --num-timesteps 3100 --verbose --progress --filter-occluded --num-agents 100 --delta-time 0.02 --num-preds 1 --pred-time 1500 --vision-pred 300 --flee-strength 50 --col-style 0 --num-runs 4 --pred-angle 180 --outfolder ~/vmodel_output/vids/back_noCol/


vmodel --num-timesteps 3100 --verbose --progress --filter-voronoi --num-agents 100 --delta-time 0.02 --num-preds 1 --pred-time 1500 --vision-pred 300 --flee-strength 50 --col-style 0 --num-runs 4 --pred-angle 90 --outfolder ~/vmodel_output/vids/side_noColVor/

vmodel --num-timesteps 3100 --verbose --progress --filter-voronoi --num-agents 100 --delta-time 0.02 --num-preds 1 --pred-time 1500 --vision-pred 300 --flee-strength 50 --col-style 0 --num-runs 4 --pred-angle 0 --outfolder ~/vmodel_output/vids/front_noColVor/

vmodel --num-timesteps 3100 --verbose --progress --filter-voronoi --num-agents 100 --delta-time 0.02 --num-preds 1 --pred-time 1500 --vision-pred 300 --flee-strength 50 --col-style 0 --num-runs 4 --pred-angle 180 --outfolder ~/vmodel_output/vids/back_noColVor/


vmodel --num-timesteps 3100 --verbose --progress --filter-occluded --num-agents 100 --delta-time 0.02 --num-preds 1 --pred-time 1500 --vision-pred 300 --flee-strength 50 --col-style 1 --num-runs 4 --pred-angle 90 --outfolder ~/vmodel_output/vids/side_noMultCol/

vmodel --num-timesteps 3100 --verbose --progress --filter-occluded --num-agents 100 --delta-time 0.02 --num-preds 1 --pred-time 1500 --vision-pred 300 --flee-strength 50 --col-style 1 --num-runs 4 --pred-angle 0 --outfolder ~/vmodel_output/vids/front_noMultCol/

vmodel --num-timesteps 3100 --verbose --progress --filter-occluded --num-agents 100 --delta-time 0.02 --num-preds 1 --pred-time 1500 --vision-pred 300 --flee-strength 50 --col-style 1 --num-runs 4 --pred-angle 180 --outfolder ~/vmodel_output/vids/back_noMultCol/


vmodel --num-timesteps 3100 --verbose --progress --filter-occluded --num-agents 100 --delta-time 0.02 --num-preds 1 --pred-time 1500 --vision-pred 300 --flee-strength 50 --col-style 1 --num-runs 4 --pred-angle 90 --outfolder ~/vmodel_output/vids/side_MultCol/

vmodel --num-timesteps 3100 --verbose --progress --filter-occluded --num-agents 100 --delta-time 0.02 --num-preds 1 --pred-time 1500 --vision-pred 300 --flee-strength 50 --col-style 1 --num-runs 4 --pred-angle 0 --outfolder ~/vmodel_output/vids/front_MultCol/

vmodel --num-timesteps 3100 --verbose --progress --filter-occluded --num-agents 100 --delta-time 0.02 --num-preds 1 --pred-time 1500 --vision-pred 300 --flee-strength 50 --col-style 1 --num-runs 4 --pred-angle 180 --outfolder ~/vmodel_output/vids/back_MultCol/
