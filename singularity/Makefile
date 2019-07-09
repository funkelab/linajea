TAG="linajea:v1.0"

TMP_FILE:=$(shell mktemp).img

.PHONY: default push
default:
	sudo singularity build $(TMP_FILE) Singularity
	cp $(TMP_FILE) $(TAG).img
	sudo rm $(TMP_FILE)
