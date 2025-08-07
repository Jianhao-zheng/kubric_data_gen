#!/bin/bash


# docker build -f docker/Blender.Dockerfile -t kubricdockerhub/blender:latest .  # build a blender image first
# docker build -f docker/Kubruntu.Dockerfile -t kubricdockerhub/kubruntu:latest .  # then build a kubric image of which base image is the blender image above



# Define the number of iterations you want
START=0
END=100  # Change this to your desired number of iterations

# Record overall start time
overall_start=$(date +%s)

for idx in $(seq $START $END); do
  # Format idx with leading zeros to be 4 digits (0001, 0002, etc.)
  padded_idx=$(printf "%04d" $idx)
  
  echo "Running job with output directory: data/$padded_idx"
  
  # Record per-job start time
  job_start=$(date +%s)
  
  docker run --rm --interactive \
    --user $(id -u):$(id -g)    \
    --volume "$(pwd):/kubric"   \
    kubricdockerhub/kubruntu    \
    /usr/bin/python3 challenges/movi/movi_def_worker.py \
      --camera=linear_movement \
      --max_camera_movement=8.0 \
      --max_motion_blur=2.0 \
    --job-dir=data/$padded_idx
  
  # Record per-job end time
  job_end=$(date +%s)
  job_duration=$((job_end - job_start))
  
  echo "Completed job $idx in ${job_duration}s"
  echo "----------------"
done

# Record overall end time
overall_end=$(date +%s)
overall_duration=$((overall_end - overall_start))

echo "All jobs completed successfully!"
echo "Total elapsed time: ${overall_duration}s"
