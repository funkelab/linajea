TAG="funkey/gp_pylp:v0.1"

.PHONY: default push
default:
	-docker rmi -f $(TAG)
	rsync -auv --delete ${GUROBI_ROOT_DIR} gurobi
	docker pull funkey/gunpowder:v0.3.2
	docker build -t $(TAG) .

push: default
	docker push $(TAG)

test: default
	docker run --rm $(TAG)
