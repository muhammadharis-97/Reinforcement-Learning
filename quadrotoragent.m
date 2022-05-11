
Ts = 0.4;
Tf = 30;


%% SET UP ENVIRONMENT

useFastRestart = true;
useGPU = true;
useParallel = true;

% Create the observation info
numObs = 9;
observationInfo = rlNumericSpec([numObs 1]);
observationInfo.Name = 'observation';

% create the action info
numAct = 4;
actionInfo = rlNumericSpec([numAct 1]);
actionInfo.UpperLimit =  ones(numAct,1);
actionInfo.LowerLimit = -ones(numAct,1);
actionInfo.Name = 'motorspeed';

%%actionInfo.LowerLimit = -ones(numAct,1);
% Environment

mdl = 'Assembly_Quadrotor';
load_system(mdl);
blk = [mdl,'/RL Agent'];
env = rlSimulinkEnv(mdl,blk,observationInfo,actionInfo);
env.ResetFcn = @(in)droneRstFcn(in);
if ~useFastRestart
  env.UseFastRestart = 'off';
end

                                 
                      
%% create critic network and actor network                      
 
hiddenLayerSize = 300;
hiddenLayerSize1 = 200;

observationPath = [
    featureInputLayer(numObs,'Normalization','none','Name','observation')
    fullyConnectedLayer(hiddenLayerSize,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(hiddenLayerSize,'Name','fc2')
    additionLayer(2,'Name','add')
    reluLayer('Name','relu2')
    fullyConnectedLayer(hiddenLayerSize,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(1,'Name','fc4')];
actionPath = [
    featureInputLayer(numAct,'Normalization','none','Name','action')
    fullyConnectedLayer(hiddenLayerSize,'Name','fc5')];


% Create the layer graph.
criticNetwork = layerGraph(observationPath);
criticNetwork = addLayers(criticNetwork,actionPath);

% Connect actionPath to observationPath.
criticNetwork = connectLayers(criticNetwork,'fc5','add/in2');

criticOptions = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);

critic = rlQValueRepresentation(criticNetwork,observationInfo,actionInfo,...
    'Observation',{'observation'},'Action',{'action'},criticOptions);

actorNetwork = [
    featureInputLayer(numObs,'Normalization','none','Name','observation')
    fullyConnectedLayer(hiddenLayerSize1,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(hiddenLayerSize1,'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(hiddenLayerSize1,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(numAct,'Name','fc4')
    tanhLayer('Name','tanh1')];

actorOptions = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);

actor = rlDeterministicActorRepresentation(actorNetwork,observationInfo,actionInfo,...
    'Observation',{'observation'},'Action',{'tanh1'},actorOptions);



agentOptions = rlDDPGAgentOptions;
agentOptions.SampleTime = Ts;
agentOptions.DiscountFactor = 0.99;
agentOptions.MiniBatchSize = 32;
agentOptions.ExperienceBufferLength = 1e6;
agentOptions.TargetSmoothFactor = 1e-3;
agentOptions.NoiseOptions.MeanAttractionConstant = 5;
agentOptions.NoiseOptions.Variance = 0.4;
agentOptions.NoiseOptions.VarianceDecayRate = 1e-5;

%% create DDPG agent

AGENT = rlDDPGAgent(actor,critic,agentOptions);

%% trainning

trainingOptions = rlTrainingOptions;
trainingOptions.MaxEpisodes = 15;
trainingOptions.MaxStepsPerEpisode = 10;
trainingOptions.ScoreAveragingWindowLength = 50;
trainingOptions.StopTrainingCriteria = 'AverageReward';
trainingOptions.StopTrainingValue = 500;
trainingOptions.SaveAgentCriteria = 'EpisodeReward';
trainingOptions.SaveAgentValue = 500;
trainingOptions.Plots = 'training-progress';
trainingOptions.Verbose = true;
if useParallel
    trainingOptions.Parallelization = 'async';
    trainingOptions.ParallelizationOptions.StepsUntilDataIsSent = 32;
    trainOpts.ParallelizationOptions.DataToSendFromWorkers = 'Experiences';
end





doTraining = true;
if doTraining    
    % Train the agent.
    trainingStats = train(AGENT,env,trainingOptions);
else
    %load('quadrotoragent.m','agent')       
end
%%curDir = pwd;
%%saveDir = 'savedAgents';
%%cd(saveDir)
%%save(['trainedAgent_3D_' datestr(now,'mm_DD_YYYY_HHMM')],'agent','trainingResults');
%%cd(curDir)
