class NanoSageChat:
    def __init__(self, max_history=3):
        self.history = []

    def add_turn(self, user_text, assistant_text):
        pass

    def clear(self):
        pass

    def get_formatted_prompt(self, current_query):
        return current_query.strip()
