import pygame as pg


P1 = 1
P2 = -1
TIE = 0

class GGP_UI:
    def __init__(self, fps, frameDim, gameDim):
        self.screen = pg.display.set_mode(frameDim)
        self.fps = fps
        self.clock = pg.time.Clock()
        self.con = True
        self.fontInit = pg.font.init()
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0 , 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.frameX = frameDim[0]
        self.frameY = frameDim[1]
        self.gameY = gameDim[0]
        self.gameX = gameDim[1]
        self.moves = []

    def setTextSettings(self, font, fontSize):
        return pg.font.SysFont(font, fontSize)

    def checkGameQuitState(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.endScreen()

    def putText(self, text, textObject, topLeftCoord, color, bold):
        self.screen.blit(textObject.render(text, bold, color), topLeftCoord)

    def endScreen(self):
        self.con = False

    def clockTick(self):
        self.clock.tick(self.fps)

    def keyPressedArray(self):
        return pg.key.get_pressed()

    def displayScreen(self):
        pg.display.flip()

    def screenFill(self, color):
        self.screen.fill(color)

    def getRectangle(self, dims, color, topLeftCoord):
        pg.draw.rect(self.screen, color, (topLeftCoord[0], topLeftCoord[1], dims[0], dims[1]))

    def getCircle(self, color, center, radius):
        pg.draw.circle(self.screen, color, center, radius)

    def getRGBArray(self):
        return pg.surfarray.array3d(self.screen)

    def initScreen(self):
        self.con = True

    def setBoard(self):
        x = 0
        textObject = self.setTextSettings("Tahoma", 20)
        self.screenFill(self.BLACK)

        for i in range(1, self.gameX + 1):
            y = 50*i
            dims = [self.frameX, 5]
            color = self.WHITE
            topLeftCoord = [x, y]
            self.getRectangle(dims, color, topLeftCoord)
        y = 0
        for i in range(1, self.gameY + 1):
            x = 50*i
            dims = [5, self.frameY]
            color = self.WHITE
            topLeftCoord = [x, y]
            self.getRectangle(dims, color, topLeftCoord)
        
        for move in self.moves:
            x = move[0][1]*50 + 20
            y = move[0][0]*50 + 15
            player = move[1]
            text = "O"
            textObject = self.setTextSettings("Tahoma", 20)
            topLeftCoord = [x, y]
            color = 0
            if player == P1:
                color = self.BLUE
            else:
                color = self.RED
            self.putText(text, textObject, topLeftCoord, color, True)

    def initBoard(self):
        self.setBoard()
        self.displayScreen()
        move = 0
        while self.con:
            self.setBoard()
            self.checkGameQuitState()
            self.displayScreen()
            self.clockTick()   

    def addMove(self, last_move, player):
        self.moves.append([last_move, player])
    
    def clearMoves(self):
        self.moves = []

# gameenv = GGP_UI(30, (350, 300), (6, 7))
# gameenv.initBoard()

