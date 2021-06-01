for i in $(eval echo {1..$1})
do
  cat job_submission.yml | sed "s/\$ID/$i/" > ./jobs-specs/job-$i.yml
done
