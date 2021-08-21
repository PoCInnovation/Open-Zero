import pygame
from typing import List, Tuple, Union
from pygame.display import set_mode
import pygame_menu

from ai_self_play import start_game

def trad_move(actions, env):
    return [env.decode(action).uci() for action in actions]
class Board:
    def __init__(self, size=[800, 800]):
        pygame.init()
        self.player_color = 'Black'
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

    def change_color(self, arg, _):
        self.player_color = arg[0][0]

    def menu(self, g):
        mytheme = pygame_menu.themes.THEME_DARK.copy()
        myimage = pygame_menu.baseimage.BaseImage(
                image_path="src/gui/assets/menu.jpg")
        mytheme.background_color = myimage
        menu = pygame_menu.Menu('Welcome to OpenZero GUI', self.width, self.height, theme=mytheme)
        menu.add.button('Play', start_game, g)
        menu.add.selector("Color", items = [('Black', 0), ('White', 0)], onchange=self.change_color)
        menu.add.button('Quit', pygame_menu.events.EXIT)

        menu.mainloop(self.screen)

    def get_piece_move(self, env_actions, piece_pose_uci):
        piece_action = []
        places = []
        i = 0
        for actions in env_actions:
            if (actions[:2] == piece_pose_uci):
                piece_action.append(actions[2:])
                places.append(i)
            i += 1
        return (piece_action, places)

    def render_from_board(self, board, direction=[]):
        for i in range(0, 8):
            for j in range(0, 8):
                self.square.update(j * self.square_size, i * self.square_size, self.square_size, self.square_size)
                pygame.draw.rect(self.board_surface, self.color_map[(j % 2 + i % 2) % 2], self.square)

        for j, i in direction:
            self.square.update((j - 1 )* self.square_size, abs(i - 8) * self.square_size, self.square_size, self.square_size)
            pygame.draw.rect(self.board_surface, pygame.Color(0, 255 , 0), self.square)

        for i in range(0, 64):
            p = board.piece_at(i)
            if not p:
                continue
            self.board_surface.blit(self.piece_map[p.symbol()], (((i % 8) * self.square_size) + self.piece_offset, ((8 - int(i / 8) - 1) * self.square_size) + self.piece_offset))
        self.screen.blit(self.board_surface, (0, 0))
        pygame.display.update()

    def envents_manager(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return (pygame.QUIT, True)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                return (event, False)
        return 0, False

    def decode_uci(self, piece_action: List[str]):
        p = []
        for i in piece_action:
            p.append((ord(i[0]) - 96, int(i[1])))
        return p

    def move_piece(self, legal_move, wanted_move) -> Union[Tuple[int, int], None]:
        i = 0
        for move in legal_move:
            if (move == wanted_move):
                return i + 1
            i += 1
        return None

    def sound_move(self):
        soundObj = pygame.mixer.Sound('src/gui/assets/chess_move.wav')
        soundObj.play()

    def wait_for_input(self, board, env_actions, env):
        decodes = []
        done = False
        trad_legal_action = trad_move(env_actions, env)
        while not done:
            event, done = self.envents_manager()
            if (event and not done) :
                x, y = event.pos
                x = int(x //self.square_size + 1)
                y = int(abs(y //self.square_size - 7) + 1)
                piece_uci = chr(int(x + 96)) + str(y)
                if (decodes != []):
                    t = self.move_piece(decodes, (x, y))
                    if (t):
                        board.push_uci(current_target + piece_uci)
                        self.sound_move()
                        return env_actions[ array_places[t - 1]]
                    decodes = []
                current_target = piece_uci
                action, array_places = self.get_piece_move(trad_legal_action, piece_uci)
                decodes = self.decode_uci(action)
                self.render_from_board(board, decodes)
