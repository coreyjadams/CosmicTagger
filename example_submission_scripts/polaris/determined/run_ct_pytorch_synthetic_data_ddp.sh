#!/bin/bash
#PBS -l select=4:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug-scaling
#PBS -l filesystems=home:grand:eagle
#PBS -A determined_eval
#PBS -V




handle_sigchld() {
      for PID in ${!PIDS[@]}; do
        if [ ! -d "/proc/$PID" ]; then
            wait $PID
            CODE=$?
            [ $CODE != 0 ] || echo >&2 INFO: process $PID exited with status $CODE;
            [ $CODE == 0 ] || echo >&2 ERROR: process $PID exited with status $CODE.  Terminating all tasks.
            unset PIDS[$PID]
            [ $CODE == 0 ] || EXIT_STATUS=$CODE && trap '' SIGCHLD && for PID in ${!PIDS[@]}; do kill $PID; done
        fi
    done
}
PIDS=()
EXIT_STATUS=0
# Enable job control so we can catch SIGCHLD
set -o monitor
trap handle_sigchld SIGCHLD

handle_preemption() {
     echo >&2 ERROR: Job $1 requested -- terminating all tasks.

      for PID in ${!PIDS[@]}; do
        if [ ! -d "/proc/$PID" ]; then
            kill -15 $PID 2> /dev/null
            CODE=$?
            [ $CODE == 0 ] || echo >&2 ERROR: kill -15 $PID exited with status $CODE.
        fi
    done
}
trap 'handle_preemption termination' SIGTERM

parallel_launch() {
     echo >&2 Launching: $*
     $* &
     PIDS[$!]=1
     echo >&2 INFO: Monitoring process $!
}
exit_with_status() {
     exit $EXIT_STATUS
}

task_num=-1; for host in $(sort -u ${PBS_NODEFILE}); do task_num=$((${task_num}+1)); parallel_launch pbs_tmrsh ${host} /grand/projects/determined_eval/bin/agents/determined-agent_0.20.0-dev0_linux_amd64/determined-agent --master-host=polaris-login-03  --master-port=8080 --resource-pool=default --container-runtime=singularity  --slot-type=cuda ;done;wait;exit_with_status