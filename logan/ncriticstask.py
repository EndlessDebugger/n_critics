class NCriticsTask():
    id = None 
    problem_statement = None 
    prompt = None
    model_response = None 
    critic_responses = None

    def __init__(self, 
                 id: int,
                 problem_statement: str):
        self.id = id
        self.problem_statement = problem_statement
        self.prompt = self.problem_statement
        self.model_response =  ""
        self.critic_responses = []
    
    def as_dict(self):
        return {"task_id": self.id, "completion": self.model_response}

