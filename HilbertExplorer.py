import numpy as np




def _binary_repr(num, width):
    """Return a binary string representation of `num` zero padded to `width`
    bits."""
    return format(num, 'b').zfill(width)



class HilbertExplorer:
    
    def __init__(self, n, l):
        """Initialize a hilbert curve with,

        Args:
            p (int): iterations to use in the hilbert curve
            n (int): number of dimensions
        """
        if l <= 0:
            raise ValueError('p must be > 0')
        if n <= 0:
            raise ValueError('n must be > 0')
        self.l = l
        self.n = n


        
    def _setCurve_(self, p, t):
        self.t = t
        self.p = p
        
        
        # maximum distance along curve
        self.max_h = 2**(self.p * self.n) - 1

        # maximum coordinate value in any dimension
        self.max_x = 2**self.p - 1
        
    def getNextT(self, t, v):
        self.v = v
        next_t = (t * (2**(self.n*self.p)-1) + v) / (2**(self.n*self.p)-1)
        return next_t
    
    def getCoord(self, t, p = None):
        if p is None:
            pass
        else:
            self.p = p
            self.max_h = 2**(self.p * self.n) - 1
            self.max_x = 2**self.p - 1
        
        self.t = t
        cur_dist = int(t * (2**(self.n*self.p)-1)) #t is in scale [0,1], dist is in scale[0, 2^(Np)-1]
        coord = self.coordinates_from_distance(cur_dist)
        return (self.coord_normalization (coord))
    
    
    def getRandomCoord(self, t, p = None):
        if p is None:
            pass
        else:
            self.p = p
        
        self.t = t
        
        # update max value
        self.max_h = 2**(self.p * self.n) - 1
        self.max_x = 2**self.p - 1
        
        dist = self.t * (2**(self.n*self.p)-1)
        dist1 = int(dist)
        dist2 = int(dist) + 1
        
        k = dist1 - dist
        
        #coord = k * coord1 + (1-k) * coord2 + random_error
        #random_error ~ N(0,1)
        coord = k * np.asarray(self.coordinates_from_distance(dist1)) + (1-k) * np.asarray(self.coordinates_from_distance(dist2)) + np.random.normal(size = self.n)
        
        return(self.coord_normalization(coord))
        
    def coord_normalization(self, coord):
        norm_coord = np.array([((coord_x / (2**(self.p-1)))-1)*self.l for coord_x in coord])
        return norm_coord
        
        

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

            
        #Example: 5 bits for each of n=3 coordinates.
        #15-bit Hilbert integer = A B C D E F G H I J K L M N O is stored as its Transpose                        ^
        #X[0] = A D G J M                    X[2] |  7
        #X[1] = B E H K N        <------->        | /X[1]
        #X[2] = C F I L O                   axes  |/
        #        high low                         0------> X[0]
        # each element in x is a p-digit value
        
        x = self._hilbert_integer_to_transpose(h)
        Z = 2 << (self.p-1)

        # Gray decode by H ^ (H/2)
        # for iteration, can be parallelized
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




