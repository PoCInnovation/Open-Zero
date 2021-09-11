import random
import ast
 
userPlayer = 'O'
boardSize = 3
numberOfSimulations = 200
 
board = [
    list('...'),
    list('...'),
    list('...')
]
 
startingPlayer = 'O'
currentPlayer = startingPlayer
 
def getBoardCopy(board):
    boardCopy = []
    
    for row in board:
        boardCopy.append( row.copy() )
    
    return boardCopy
 
def hasMovesLeft(board):
    for y in range(boardSize):
        for x in range(boardSize):
            if board[y][x] == '.':
                return True
    
    return False
 
def getNextMoves(currentBoard, player):
    nextMoves = []
    
    for y in range(boardSize):
        for x in range(boardSize):
            if currentBoard[y][x] == '.':
                boardCopy = getBoardCopy(currentBoard)
                boardCopy[y][x] = player
                nextMoves.append(boardCopy)
    
    return nextMoves
 
def hasWon(currentBoard, player):
    winningSet = [player for _ in range(boardSize)]
    
    for row in currentBoard:
        if row == winningSet:
            return True
    
    for y in range(len(currentBoard)):
        column = [currentBoard[index][y] for index in range(boardSize)]
        
        if column == winningSet:
            return True
    
    diagonal1 = []
    diagonal2 = []
    for index in range(len(currentBoard)):
        diagonal1.append(currentBoard[index][index])
        diagonal2.append(currentBoard[index][boardSize - index - 1])
    
    if diagonal1 == winningSet or diagonal2 == winningSet:
        return True
    
    return False
 
def getNextPlayer(currentPlayer):
    if currentPlayer == 'X':
        return 'O'
    
    return 'X'
 
def getBestNextMove(currentBoard, currentPlayer):
    evaluations = {}
    
    for generation in range(numberOfSimulations):
        player = currentPlayer
        boardCopy = getBoardCopy(currentBoard)
        
        simulationMoves = []
        nextMoves = getNextMoves(boardCopy, player)
        
        score = boardSize * boardSize
        
        while nextMoves != []:
            roll = random.randint(1, len(nextMoves)) - 1
            boardCopy = nextMoves[roll]
            
            simulationMoves.append(boardCopy)
            
            if hasWon(boardCopy, player):
                break
            
            score -= 1
            
            player = getNextPlayer(player)
            nextMoves = getNextMoves(boardCopy, player)
        
        firstMove = simulationMoves[0]
        lastMove = simulationMoves[-1]
        
        firstMoveKey = repr(firstMove)
        
        if player == userPlayer and hasWon(boardCopy, player):
            score *= -1
        
        if firstMoveKey in evaluations:
            evaluations[firstMoveKey] += score
        else:
            evaluations[firstMoveKey] = score
    
    bestMove = []
    highestScore = 0
    firstRound = True
    
    for move, score in evaluations.items():
        if firstRound or score > highestScore:
            highestScore = score
            bestMove = ast.literal_eval(move)
            firstRound = False
    
    return bestMove
 
def printBoard(board):
    firstRow = True
    
    for index in range(boardSize):
        if firstRow:
            print('  012')
            firstRow = False
            
        print( str(index) + ' ' + ''.join(board[index]) )
 
def getPlayerMove(board, currentPlayer):
    isMoveValid = False
    while isMoveValid == False:
        print('')
        userMove = input('X,Y? ')
        userX, userY = map(int, userMove.split(','))
        
        if board[userY][userX] == '.':
            isMoveValid = True
    
    board[userY][userX] = currentPlayer
    return board
 
printBoard(board)
 
while hasMovesLeft(board):
    if currentPlayer == userPlayer:
        board = getPlayerMove(board, currentPlayer)
    else :
        board = getBestNextMove(board, currentPlayer)
    
    print('')
    printBoard(board)
    
    if hasWon(board, currentPlayer):
        print('Player ' + currentPlayer + ' has won!')
        break
    
    currentPlayer = getNextPlayer(currentPlayer)