import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import KFold

class denseAutoencoder1D(tf.keras.Model):
    def __init__(self, x_train, x_val, x_test=None):
        super().__init__()
        self.x_train=x_train
        self.x_val=x_val
        self.x_test=x_test

        self.input_size=x_train.shape[1]
        self.output_size=self.input_size
        self.latent_nodes=0
        self.encoder_decoder_nodes=[]
        self.l_r=0.001
        self.activation='tanh'
        self.best_params=None
        self.latent_nodes_range=None
    
    def Initialise(self, encoder_decoder_nodes, latent_nodes, activation=None, l_r=None, loss='mean_squared_error'):
        self.latent_nodes=latent_nodes
        if activation==None:
            activation=self.activation
        if l_r!=None:
            self.l_r=l_r

        if type(encoder_decoder_nodes)==int:
            encoder_decoder_nodes=[encoder_decoder_nodes]
        self.encoder_decoder_nodes=encoder_decoder_nodes

        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.layers.Input(shape=(self.input_size,)))   
        for n_nodes in self.encoder_decoder_nodes:
            if n_nodes>0:
                self.encoder.add(tf.keras.layers.Dense(n_nodes, activation=self.activation))
        self.encoder.add(tf.keras.layers.Dense(self.latent_nodes, activation=self.activation))

        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.layers.Input(shape=(self.latent_nodes,)))        
        for n_nodes in reversed(self.encoder_decoder_nodes):
            if n_nodes>0:
                self.decoder.add(tf.keras.layers.Dense(n_nodes, activation=self.activation))
        self.decoder.add(tf.keras.layers.Dense(self.output_size, activation='linear'))

        self.compile(optimizer=tf.keras.optimizers.Adam(self.l_r), loss='mse')
    
    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    def reconstruct(self, input=None):
        if input==None:
            input=self.x_test
        return self.predict(input)

    def predictLatent(self, input=None):
        if input==None:
            input=x_test
        if input==None:
            print("No test data to predict from.")
            return 0
        return self.encoder.predict(input)

    def findBestHP(self, num_folds = 3, n_initial_points=10, n_optimize_points=100, epochs=250, patience=50, latent_nodes_range=(4, 256), encoded_nodes_range=[(4, 1024)], l_r_range=(0.0001, 0.1), activations=['relu', 'sigmoid', 'tanh']):
        X_train=np.concatenate((self.x_train, self.x_val), axis=0)
        self.latent_nodes_range = latent_nodes_range
        # Define the search space for the hyperparameters
        space=[]
        if type(l_r_range)==tuple:
            space+=[Real(l_r_range[0], l_r_range[1], "log-uniform", name='learning_rate')]
        if type(activations)==list:
            space += [Categorical(activations, name='activation')]
        if type(latent_nodes_range)!=int:
            space += [Integer(latent_space_range[0], latent_space_range[1], name="Latend Space Size")]
        if type(encoded_nodes_range)==tuple:
            encoded_nodes_range=[encoded_nodes_range]
        for i in range(len(encoded_nodes_range)):
            space += [Integer(encoded_nodes_range[i][0], encoded_nodes_range[i][1], name='n_neurons_'+str(i))]

        def objective(params):
            k=0
            if type(l_r_range)==tuple:
                l_r=params[k]
                k+=1
            else:
                l_r=l_r_range
            
            if type(activations)==list:
                activation=params[k]
                k+=1
            else:
                activation = activations
                
            if type(latent_nodes_range)!=int:
                latent_nodes=params[k]
                k+=1
            else: 
                latent_nodes=latent_nodes_range
                
                
            string = "Learning Rate: %f, Activation: %s, Latent Space Size: %d, " % (l_r, activation, latent_nodes)
            for i in range(k, len(params)-1):
                string += "No. Nodes in Layer %d: %d, " % (i-(k-1), params[i])
            string += "No. Nodes in Layer %d: %d." % (len(params)-k, params[-1])
            print(string)
            
            encoder_decoder_nodes=[]
            for i in range(k, len(params)):
                encoder_decoder_nodes.append(params[i])
            for nodes in encoder_decoder_nodes:
                if nodes!=0 and nodes<latent_nodes:
                    return 1e10
            
            test_scores=[]
            kf=KFold(num_folds, shuffle=True)
            for train_index, val_index in kf.split(x_train):
                X_train, X_val = x_train[train_index], x_train[val_index]
                attempt_ae=denseAutoencoder1D(X_train, X_val)
                attempt_ae.Initialise(encoder_decoder_nodes, latent_nodes, activation=activation, l_r=l_r)
                try:
                    attempt_ae.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=epochs, verbose=0)
                except:
                    return 1e10
                # Evaluate the model on the test set
                x_pred = attempt_ae.predict(x_test)
                score = mean_squared_error(x_test, x_pred)
                test_scores.append(score)
            del attempt_ae
            mean_val_score = np.mean(test_scores)
            print("Mean Score of KFolds={:.2e}".format(mean_val_score))
            return mean_val_score

        # Use bayesian optimization to find the optimal hyperparameters
        res = gp_minimize(objective, space, n_calls=(n_optimize_points+n_initial_points), random_state=0, n_initial_points=n_initial_points)
        print(res.x)
        self.best_params=res.x
        return res


    def InitialiseBestHP(self):
        if self.best_params==None:
            print("You need to call findBestHP() first.")
            return
        k=0
        l_r=self.best_params[k]
        k+=1
        activation=self.best_params[k]
        k+=1
        if type(self.latent_nodes_range)!=int:
            latent_nodes=self.best_params[k]
            k+=1
        else: 
            latent_nodes=self.latent_nodes_range


        string = "Learning Rate: %f, Activation: %s, Latent Space Size: %d, " % (l_r, activation, latent_nodes)
        for i in range(k, len(self.best_params)-1):
            string += "No. Nodes in Layer %d: %d, " % (i-(k-1), self.best_params[i])
        string += "No. Nodes in Layer %d: %d." % (len(self.best_params)-k, self.best_params[-1])
        print(string)

        encoder_decoder_nodes=[]
        for i in range(k, len(self.best_params)):
            encoder_decoder_nodes.append(self.best_params[i])
                
        self.Initialise(encoder_decoder_nodes=encoder_decoder_nodes, latent_nodes=latent_nodes, activation=activation, l_r=l_r)
    
