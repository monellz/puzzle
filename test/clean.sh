#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)

rm ${SCRIPT_DIR}/*.ll
rm ${SCRIPT_DIR}/*.s
rm ${SCRIPT_DIR}/*.o
rm ${SCRIPT_DIR}/*.bc
rm ${SCRIPT_DIR}/*.out
