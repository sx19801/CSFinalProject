from mesa import Agent
from mesa import Model

class PreyAgent(Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, model, q_table):
        super().__init__(unique_id, model)
        self.berries = 0

    def collect_berry(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        for agent in cellmates:
            if isinstance(agent, BerryAgent):
                self.model.grid.remove_agent(agent)
                self.berries += 1
    
    def Qmove(self) -> None:
        

    def move(self) -> None:
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False
        )
        #possible_steps = [cell for cell in possible_steps if not self.model.grid.is_cell_empty(cell)]
        
        
        if possible_steps:
            new_position = self.random.choice(possible_steps)
            self.model.grid.move_agent(self, new_position)
            
        possible_steps_updated = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False
        )
        print(f"possible steps: {possible_steps_updated}")
    
    def step(self) -> None:
        self.move()
        
        self.collect_berry()
        
class BerryAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
            
class Obstacle(Agent):
    def __init__(self) -> None:
        super().__init__(unique_id = None, model=None)
        
    def draw(self):
        portrayal = {"Shape": "rect",
                     "Color": "red",
                     "Filled": "true",
                     "Layer": 0,
                     "w": 1,
                     "h": 1}
        return portrayal