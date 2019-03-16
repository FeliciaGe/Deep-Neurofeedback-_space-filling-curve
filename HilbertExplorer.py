import numpy as np
from itertools import permutations 


class HilbertExplorer:
    
    p = 1   # order of Hilbert Curve
    t = 0.5 # current position on the curve
    l = 1   # side length of space
    v = 1   # initial velocity
    
    
    def __init__(self, n, l = None, rho = None):
        '''Intialize Hiblert tExplorer with:
        Args:
            n: Dimension of explored spapce
            l: side length of space 
            Size of Explored Space: [-l, l]^ N
            rho: number of points return between on a segment
        '''

        if l is None: 
            l = 1
        elif l <= 0:
            raise ValueError('l must > 0')
        
        if n <= 0:
            raise ValueError('N must be > 0')
            
        if rho is None:
            self.rho = 1
        else:
            self.rho = rho

        self.n = n
        self.l = l
        
        #default permutation is original coordinate
        self.Perm = list(range(self.n)) 
        
        # maximum distance along curve
        self.max_h = 2**(self.p * self.n) - 1

        # maximum coordinate value in any dimension
        self.max_x = 2**self.p - 1
        
        
    '''    
    def printPerm(self):
        permList = list(permutations(range(self.n)))
        print(permList)
    '''
    
    #input an index to get order of permutation
    def setPermIndex(self, index):
        permList = list(permutations(range(self.n)))
        self.Perm = np.array(permList[index])
        
    #input the Permutation List
    def setPermIndex2(self, permIndexList):
        self.Perm = np.array(permIndexList)

    def getPermCoord(self, coord):
        new_coord = [coord[i] for i in self.Perm]
        return np.array(new_coord)
        
    def setT(self, t):
        self.t = t
        self.dist = t * (2 ** (self.n * self.p) - 1)
    
        
    def getCoord(self, t, p = None):
        if p is None:
            pass
        else:
            self.p = p
        
        self.t = t
        self.dist = t * (2 ** (self.n * self.p) - 1)
            
        # update max value
        self.max_h = 2**(self.p * self.n) - 1
        self.max_x = 2**self.p - 1
        
        cur_dist = int(self.t * (2**(self.n*self.p)-1)) #t is in scale [0,1], dist is in scale[0, 2^(Np)-1]
        coord = self.coordinates_from_distance(cur_dist)
        norm_coord = self.coord_normalization(coord)
        perm_coord = self.getPermCoord(list(norm_coord))
        return (perm_coord)
    
    def setP(self, p):
        self.p = p
        # maximum distance along curve
        self.max_h = 2**(self.p * self.n) - 1

        # maximum coordinate value in any dimension
        self.max_x = 2**self.p - 1
        
    def getNextCoord(self, v, t):
        dist = t * (2 ** (self.n * self.p) - 1)
        next_dist = int(dist + v)
        return (self.getPermCoord(self.coord_normalization(self.coordinates_from_distance(next_dist))))

    def getNextCoordFromDist(self, v, dist):
        next_dist = int(dist + v)
        return (self.getPermCoord(self.coord_normalization(self.coordinates_from_distance(next_dist))))


    def updateDist(self, v):
        self.dist = int(self.dist + v)
    
    def getRandomCoord(self, t, p = None):
        if p is None:
            pass
        else:
            self.p = p
        
        self.t = t
        self.dist = t * (2 ** (self.n * self.p) - 1)
        
        # update max value
        self.max_h = 2**(self.p * self.n) - 1
        self.max_x = 2**self.p - 1
        
        dist = self.t * (2**(self.n*self.p)-1)
        dist1 = int(dist)
        dist2 = int(dist) + 1
        
        k = dist1 - dist
        coord = k * np.asarray(self.coordinates_from_distance(dist1)) + (1-k) * np.asarray(self.coordinates_from_distance(dist2)) + np.random.normal(size = self.n)
        return (self.getPermCoord(self.coord_normalization(coord)))

    def getNextT(self, v = None, p = None, t = None):
        #input v, get next T
        
        if v is None:
            pass
        else:
            self.v = v
        
        if p is None:
            pass
        else:
            self.p = p
            
            
        if t is None:
            pass
        else:
            self.t = t
            self.dist = t * (2 ** (self.n * self.p) - 1)
    
        self.v = v
        
        #next_t = (self.t * (2**(self.n*self.p)-1) + self.v) / (2**(self.n*self.p)-1)
        next_t = self.t + self.v / (2**(self.n*self.p)-1)
        return next_t


    def _hilbert_integer_to_transpose(self, h):
        """Store a hilbert integer (`h`) as its transpose (`x`).

        Args:
            h (int): integer distance along hilbert curve

        Returns:
            x (list): transpose of h
                      (n components with values between 0 and 2**p-1)
        """
        h_bit_str = _binary_repr(h, self.p*self.n)
        x = [int(h_bit_str[i::self.n], 2) for i in range(self.n)]
        return x

    def _transpose_to_hilbert_integer(self, x):
        """Restore a hilbert integer (`h`) from its transpose (`x`).

        Args:
            x (list): transpose of h
                      (n components with values between 0 and 2**p-1)

        Returns:
            h (int): integer distance along hilbert curve
        """
        x_bit_str = [_binary_repr(x[i], self.p) for i in range(self.n)]
        h = int(''.join([y[i] for i in range(self.p) for y in x_bit_str]), 2)
        return h
    
    def coord_normalization(self, coord):
        norm_coord = np.array([((coord_x / (2**(self.p-1)))-1)*self.l for coord_x in coord])
        return norm_coord
        

    def coordinates_from_distance(self, h):
        """Return the coordinates for a given hilbert distance.

        Args:
            h (int): integer distance along hilbert curve

        Returns:
            x (list): transpose of h
                      (n components with values between 0 and 2**p-1)
        """
        if h > self.max_h:
            raise ValueError('h={} is greater than 2**(p*N)-1={}'.format(h, self.max_h))
        if h < 0:
            raise ValueError('h={} but must be > 0'.format(h))

        x = self._hilbert_integer_to_transpose(h)
        Z = 2 << (self.p-1)

        # Gray decode by H ^ (H/2)
        t = x[self.n-1] >> 1
        for i in range(self.n-1, 0, -1):
            x[i] ^= x[i-1]
        x[0] ^= t

        # Undo excess work
        Q = 2
        while Q != Z:
            P = Q - 1
            for i in range(self.n-1, -1, -1):
                if x[i] & Q:
                    # invert
                    x[0] ^= P
                else:
                    # exchange
                    t = (x[0] ^ x[i]) & P
                    x[0] ^= t
                    x[i] ^= t
            Q <<= 1

        # done
        return x

    def getCoordList(self, t, p = None, rho = None):
        if rho is None:
            pass
        else:
            self.rho = rho
            
        if p is None:
            pass
        else:
            self.p = p
        
        self.t = t
        self.dist = t * (2 ** (self.n * self.p) - 1)
        
        # update max value
        self.max_h = 2**(self.p * self.n) - 1
        self.max_x = 2**self.p - 1
        
        dist = self.t * (2**(self.n*self.p)-1)
        #dist1 = int(dist)
        #dist2 = int(dist) + 2
        
        temp = self.rho - 1
        k_list = np.arange(0, 1+1/temp, 1/temp)
        
        #coord1 = np.asarray(self.coord_normalization(self.coordinates_from_distance(dist1)))
        #coord2 = np.asarray(self.coord_normalization(self.coordinates_from_distance(dist2)))
        coord1 = self.getCoord(t, self.p)
        
        #next_t = (t * (2**(self.n*self.p)-1) + 2) / (2**(self.n*self.p)-1)
        #print(t)
        #print(float(next_t))
        coord2 = self.getNextCoord(1, t)
        
        coord_list = [[ k * coord1[i] + (1-k) * coord2[i] for i in range(self.n)] for k in k_list] 
        
        perm_list = [self.getPermCoord(coord) for coord in coord_list]

        return(coord_list)
    
    
    def distance_from_coordinates(self, x_in):
        """Return the hilbert distance for a given set of coordinates.

        Args:
            x_in (list): transpose of h
                         (n components with values between 0 and 2**p-1)

        Returns:
            h (int): integer distance along hilbert curve
        """
        x = list(x_in)
        if len(x) != self.n:
            raise ValueError('x={} must have N={} dimensions'.format(x, self.n))

        if any(elx > self.max_x for elx in x):
            raise ValueError(
                'invalid coordinate input x={}.  one or more dimensions have a '
                'value greater than 2**p-1={}'.format(x, self.max_x))

        if any(elx < 0 for elx in x):
            raise ValueError(
                'invalid coordinate input x={}.  one or more dimensions have a '
                'value less than 0'.format(x))

        M = 1 << (self.p - 1)

        # Inverse undo excess work
        Q = M
        while Q > 1:
            P = Q - 1
            for i in range(self.n):
                if x[i] & Q:
                    x[0] ^= P
                else:
                    t = (x[0] ^ x[i]) & P
                    x[0] ^= t
                    x[i] ^= t
            Q >>= 1

        # Gray encode
        for i in range(1, self.n):
            x[i] ^= x[i-1]
        t = 0
        Q = M
        while Q > 1:
            if x[self.n-1] & Q:
                t ^= Q - 1
            Q >>= 1
        for i in range(self.n):
            x[i] ^= t

        h = self._transpose_to_hilbert_integer(x)
        return h


        


def _binary_repr(num, width):
    """Return a binary string representation of `num` zero padded to `width`
    bits."""
    return format(num, 'b').zfill(width)



