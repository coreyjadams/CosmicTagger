#!/bin/bash
if [ $PMIX_RANK -eq 0 ]
then
  $*
else
  $* >& /dev/null
fi
