IMAGE  := portfolio
PORT   := 8080

build:
	docker build -t $(IMAGE) .

run:
	docker run -d --name $(IMAGE) -p $(PORT):8080 $(IMAGE)

stop:
	docker stop $(IMAGE)

rm:
	docker rm $(IMAGE)

down: stop rm

rebuild: down build run

logs:
	docker logs -f $(IMAGE)

shell:
	docker exec -it $(IMAGE) sh

clean:
	docker rmi $(IMAGE)

.PHONY: build run stop rm down rebuild logs shell clean
