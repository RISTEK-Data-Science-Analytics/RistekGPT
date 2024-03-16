import threading
from queue import Queue
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import HuggingFaceEndpoint

# Custom callback handler that puts tokens into a queue
class QueueStreamingCallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self):
        super().__init__()
        self.output_queue = Queue()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Put new LLM token into the queue."""
        self.output_queue.put(token)

    def get_queue(self):
        return self.output_queue

# Function to start LLM generation in a separate thread
def start_llm_generation(llm, input_text, callback_handler):
    def _target():
        llm(input_text, callbacks=[callback_handler])

    threading.Thread(target=_target).start()
    return callback_handler.get_queue()

# Generator function to read from the queue and yield tokens
def stream_tokens(queue):
    while True:
        token = queue.get()
        if token is None or token=="<eos>":  # Use None as a signal for the end of the stream
            break
        yield token