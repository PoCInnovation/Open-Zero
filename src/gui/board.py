import pygame

class Board:
    def __init__(self, size=[800, 800]):
        pygame.init()
        self.width = size[0]
        self.height = size[1]
        self.screen_size = size
        self.screen = pygame.display.set_mode(self.screen_size)
        self.board_size = self.width
        self.board_surface = pygame.Surface([self.board_size, self.board_size])
        self.square_size = self.board_size / 8
        self.square = pygame.Rect(0, 0, self.square_size, self.square_size)
        self.light_square_color = pygame.Color(210, 210, 210)
        self.dark_square_color = pygame.Color(40, 40, 40)
        self.color_map = [self.light_square_color, self.dark_square_color]
        self.font = pygame.font.SysFont('Arial', 70)
        self.piece_map = {}
        scale = (90, 90)
        self.piece_offset = (self.square_size - scale[0]) / 2
        self.piece_map['P'] = pygame.transform.scale(pygame.image.load('src/gui/assets/w_pawn_2x_ns.png'), scale)
        self.piece_map['K'] = pygame.transform.scale(pygame.image.load('src/gui/assets/w_king_2x_ns.png'), scale)
        self.piece_map['Q'] = pygame.transform.scale(pygame.image.load('src/gui/assets/w_queen_2x_ns.png'), scale)
        self.piece_map['R'] = pygame.transform.scale(pygame.image.load('src/gui/assets/w_rook_2x_ns.png'), scale)
        self.piece_map['B'] = pygame.transform.scale(pygame.image.load('src/gui/assets/w_bishop_2x_ns.png'), scale)
        self.piece_map['N'] = pygame.transform.scale(pygame.image.load('src/gui/assets/w_knight_2x_ns.png'), scale)

        self.piece_map['p'] = pygame.transform.scale(pygame.image.load('src/gui/assets/b_pawn_2x_ns.png'), scale)
        self.piece_map['k'] = pygame.transform.scale(pygame.image.load('src/gui/assets/b_king_2x_ns.png'), scale)
        self.piece_map['q'] = pygame.transform.scale(pygame.image.load('src/gui/assets/b_queen_2x_ns.png'), scale)
        self.piece_map['r'] = pygame.transform.scale(pygame.image.load('src/gui/assets/b_rook_2x_ns.png'), scale)
        self.piece_map['b'] = pygame.transform.scale(pygame.image.load('src/gui/assets/b_bishop_2x_ns.png'), scale)
        self.piece_map['n'] = pygame.transform.scale(pygame.image.load('src/gui/assets/b_knight_2x_ns.png'), scale)


    def render(self):
        pass

    def render_from_board(self, board):
        for i in range(0, 8):
            for j in range(0, 8):
                self.square.update(j * self.square_size, i * self.square_size, self.square_size, self.square_size)
                pygame.draw.rect(self.board_surface, self.color_map[(j % 2 + i % 2) % 2], self.square)
        
        for i in range(0, 64):
            p = board.piece_at(i)
            if not p:
                continue
            self.board_surface.blit(self.piece_map[p.symbol()], (((8 - (i % 8) - 1) * self.square_size) + self.piece_offset, ((8 - int(i / 8) - 1) * self.square_size) + self.piece_offset))

        self.screen.blit(self.board_surface, (0, 0))
        pygame.display.update()
    
    def wait_for_input(self):
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    return True
                
