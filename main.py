# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import random
import pygame
import os
import neat

pygame.font.init()

PIPE_IMG = pygame.image.load(os.path.join("imgs", "pipe.png"))
BASE_IMG = pygame.image.load(os.path.join("imgs", "base.png"))
BG_IMG = pygame.image.load(os.path.join("imgs", "bg.png"))
BIRD_IMGS = [pygame.image.load(os.path.join("imgs", "bird1.png")),
             pygame.image.load(os.path.join("imgs", "bird2.png")),
             pygame.image.load(os.path.join("imgs", "bird3.png"))]
WIN_WIDTH = BG_IMG.get_width()
WIN_HEIGHT = BG_IMG.get_height()

STAT_FONT = pygame.font.SysFont("comicsans", 50)


class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1
        d = self.vel * self.tick_count + 1.5 * self.tick_count ** 2

        if d >= 16:
            d = 16

        if d < 0:
            d -= 2
        self.y = self.y + d

        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count += 1

        # For animation of bird, loop through three images
        if self.img_count <= self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count <= self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
        elif self.img_count <= self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
        elif self.img_count <= self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME * 4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME * 2

        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(int(self.x), int(self.y))).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.height = 100
        self.gap = 150
        self.passed = 0

        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG
        self.set_height()

    def set_height(self):
        self.height = random.randrange(round(WIN_HEIGHT * 0.05), round(WIN_HEIGHT * 0.60))
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.gap

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (int(self.x), int(self.top)))
        win.blit(self.PIPE_BOTTOM, (int(self.x), int(self.bottom)))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (int(self.x - bird.x), int(self.top - bird.y))
        bottom_offset = (int(self.x - bird.x), int(self.bottom - bird.y))

        top_collide = bird_mask.overlap(top_mask, top_offset)
        bottom_collide = bird_mask.overlap(bottom_mask, bottom_offset)

        if top_collide or bottom_collide:
            return True

        return False


class Base:
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (int(self.x1), int(self.y)))
        win.blit(self.IMG, (int(self.x2), int(self.y)))


def draw_window(win, birds, base, pipes, score):
    win.blit(BG_IMG, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    text = STAT_FONT.render("Score:" + str(score), 1, (255, 255, 255))
    win.blit(text, (0, 0))

    base.draw(win)

    for bird in birds:
        bird.draw(win)

    pygame.display.update()


def main(genomes, config):
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    pipe_height = WIN_HEIGHT * 0.55
    base_height = WIN_HEIGHT * 0.85
    score = 0

    nets = []
    ge = []
    birds = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(WIN_WIDTH * 0.1, WIN_HEIGHT * 0.4))
        g.fitness = 0
        ge.append(g)

    base = Base(base_height)
    pipes = [Pipe(pipe_height)]

    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        if len(birds) <= 0:
            break

        for idx, bird in enumerate(birds):
            bird.move()
            ge[idx].fitness += 0.1

            output = nets[idx].activate((bird.y,
                                         abs(bird.y - pipes[pipes[0].passed].height),
                                         abs(bird.y - pipes[pipes[0].passed].bottom)))
            if output[0] > 0.5:
                bird.jump()

        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()
            for idx, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[idx].fitness -= 1
                    birds.pop(idx)
                    nets.pop(idx)
                    ge.pop(idx)

                if not pipe.passed and bird.x > pipe.x + pipe.PIPE_TOP.get_width():
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5
            pipes.append(Pipe(pipe_height))

        for r in rem:
            pipes.remove(r)

        for idx, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= base_height or bird.y < 0:
                birds.pop(idx)
                nets.pop(idx)
                ge.pop(idx)
        base.move()
        draw_window(win, birds, base, pipes, score)


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(main, 50)
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat_config.txt")
    run(config_path)
