# Larcv Training Utils

This repository contains some core, skeleton code to get a network up and running using larcv.  It's designed for larcv3.

It has a master branch with just some core function placeholders, and IO implementation.  It has a `torch` and `tf` branch with more functions filled in for tensorflow and torch.  The intention is to fork this repo for your own purposes so you can fill in your own model and various details, without having to worry too much about opening and closing IO utilities, starting data loading, keeping io in sync, etc.

This repo also contains tools for distributed learning, including learning rate schedules based on epochs and performing allreduce on metrics, etc.  Horovod and MPI4PY are the communication tools.

The master branch has ability to run io tests on files, so if you only want to perform IO benchmarks on a larcv dataset, fork this repository and keep the master branch going.  Then, you can run IO tests even in distributed mode.