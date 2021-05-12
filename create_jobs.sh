for i in $(eval echo {1..$1})
do
  cat deployment.yml | sed "s/\$ID/$i/" > ./jobs-specs/job-$i.yml
done
