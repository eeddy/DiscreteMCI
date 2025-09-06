import pygame
import random
import numpy as np 
import time
import pickle

EVALUATION = True

class Evaluation:
    def __init__(self, num_trials=4):
        self.classes = ['Close', 'Extension', 'Flexion', 'Open', 'Pinch'] 
        self.trials = [0, 0, 0, 0, 0]
        if EVALUATION:
            self.num_trials = 25
        else:
            self.num_trials = num_trials
        
        self.next_trial = True
        self.time = -1
        self.trial = 0 
        self.spawn_time = random.uniform(1, 2)
        self.pause = False
        self.log = {
            'timestamp': [],
            'target': [],
            'prediction': [],
            'trial': [],
        }
        random.seed(time.time())

    def start_game(self):
        pygame.init()

        screen = pygame.display.set_mode([500, 500])

        running = True
        while running:
            screen.fill('white')

            if self.pause:
                print("Press Enter to continue...")
                while self.pause:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_RETURN:
                                self.pause = False


            if self.next_trial:
                id = random.randint(0, len(self.classes) - 1)
                if EVALUATION:
                    while min(self.trials) < self.trials[id]:
                        id = random.randint(0, len(self.classes) - 1)
                else:
                    while self.trials[id] == self.num_trials:
                        id = random.randint(0, len(self.classes) - 1)
                self.next_trial = False
            
            prediction = -1 

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    prediction = event.key
                    if ((event.key == pygame.K_f and id == self.classes.index('Flexion'))
                        or (event.key == pygame.K_e and id == self.classes.index('Extension'))
                        or (event.key == pygame.K_o and id == self.classes.index('Open'))
                        or (event.key == pygame.K_c and id == self.classes.index('Close'))
                        or (event.key == pygame.K_p and id == self.classes.index('Pinch'))
                        ):
                        self.trials[id] += 1 
                        self.next_trial = True
                        self.time = time.time()
                        self.trial += 1
                        self.spawn_time = random.uniform(1, 2)

                        if sum(self.trials) % 25 == 0 and EVALUATION and sum(self.trials) != 0:
                            self.pause = True 
                        else: 
                            self.pause = False

            font = pygame.font.SysFont(None, 36)
            if EVALUATION:
                text = str(sum(self.trials)) + '/100'
            else:
                text = str(sum(self.trials)) + '/25'
            text_surface = font.render(text, True, (0, 0, 0))
            text_rect = text_surface.get_rect()
            text_rect.topright = (500 - 10, 10)
            screen.blit(text_surface, text_rect)

            if time.time() - self.time >= self.spawn_time:
                label = self.classes[id]
                font = pygame.font.Font(None, 50)
                text = font.render(label, True, (0, 0, 0))
                screen.blit(text, (200, 10))
                img = pygame.image.load('Other/Images/' + self.classes[id] + '.png')
                screen.blit(img,(75,65))
                self.log['trial'].append(self.trial)
            else:
                self.log['trial'].append(-1)

            self.log['timestamp'].append(time.time())
            self.log['target'].append(id)
            self.log['prediction'].append(prediction)

            pygame.display.flip()

            if sum(self.trials) == len(self.trials) * self.num_trials:
                running = False

        pygame.quit()

eval = Evaluation()
eval.start_game()