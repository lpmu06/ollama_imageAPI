./bin/ollama serve &

pid=$!

sleep 5


echo "Pulling llava:7b model"
ollama pull llava:7b


wait $pid