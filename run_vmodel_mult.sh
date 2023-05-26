#!/bin/bash


vmodel --num-timesteps 1200 --verbose --progress --filter-occluded --num-agents 100 --delta-time 0.05 --num-preds 1 --pred-time 500 --vision-pred 300 --num-runs 1

vmodel --num-timesteps 12000 --verbose --progress --filter-occluded --num-agents 100 --delta-time 0.05 --num-preds 5 --pred-time 500 --vision-pred 180 --num-runs 1

vmodel --num-timesteps 12000 --verbose --progress --filter-occluded --num-agents 100 --delta-time 0.05 --num-preds 5 --pred-time 500 --vision-pred 300 --num-runs 1


vmodel --num-timesteps 12000 --verbose --progress --filter-occluded --num-agents 100 --delta-time 0.05 --num-preds 5 --pred-time 500 --repradius-pred 5 --num-runs 1

vmodel --num-timesteps 12000 --verbose --progress --filter-occluded --num-agents 100 --delta-time 0.05 --num-preds 5 --pred-time 500 --repradius-pred 10 --num-runs 1

vmodel --num-timesteps 12000 --verbose --progress --filter-occluded --num-agents 100 --delta-time 0.05 --num-preds 5 --pred-time 500 --repradius-pred 25 --num-runs 1




vmodel --num-timesteps 12000 --verbose --progress --filter-occluded --num-agents 100 --delta-time 0.05 --num-preds 5 --pred-time 500 --dphi 0.01 --num-runs 1

vmodel --num-timesteps 12000 --verbose --progress --filter-occluded --num-agents 100 --delta-time 0.05 --num-preds 5 --pred-time 500 --dphi 1 --num-runs 1





vmodel --num-timesteps 12000 --verbose --progress --filter-occluded --num-agents 100 --delta-time 0.05 --num-preds 5 --pred-time 500 --repulsion-pred 0.5 --num-runs 1

vmodel --num-timesteps 12000 --verbose --progress --filter-occluded --num-agents 100 --delta-time 0.05 --num-preds 5 --pred-time 500 --repulsion-pred 2 --num-runs 1

vmodel --num-timesteps 12000 --verbose --progress --filter-occluded --num-agents 100 --delta-time 0.05 --num-preds 5 --pred-time 500 --repulsion-pred 5 --num-runs 1



