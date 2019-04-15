%% dot and box games implemented considering symmetry
line_no=12;
state=zeros(1,line_no);
figure(1)
axis([0,2,0,2]);
count=0;
stats_x=zeros(1,15);
state_y=zeros(3,15);
opponent_type=input('Enter R to play random opponent\nAEnter H to play with human opponent:','s');
%colormap gray
%cimage=[0 0;0 0];
set(gca,'XTick',0:2,'YTick',0:2);
%set(gca,'LineWidth',4)

grid on
hold on
%imagesc(0:2,0:2,cimage)
i=0;% turn determinant for the player
m=0;% episode determinant
eps=.1;
Q=zeros(2^line_no,line_no);
Qaccess=zeros(2^line_no,line_no);
cum_episode=0;
cum_episode_R=0;
cum_episode_H=0;
win=0;
lose=0;
draw=0;
wine=0;
losee=0;
drawe=0;
load('sym_Qval.mat','Qaccess','Q','stats_x','stats_y');
load('sym_Qvalue.mat','cum_episode','count','cum_episode_H','win','lose','draw');
Qsupdate=zeros(5,line_no); % matrix for keeping the state action value to be updated
Qacupdate=zeros(5,line_no);
acupdate=zeros(5,line_no)
cum_reward=zeros(5,line_no) % matrix for keeping reward gained at each time step

move_agent=1; % indicates agent move no.
tot_reward=0; % total reward in an episode
episode=1;
boxcount_agent=0;
boxcount_opponent=0;
winH=0;
loseH=0;
drawH=0;
while(episode<2)
    
    
    i=i+1;
    m=m+1;
   % pause(3);
    if (mod((i+episode),2)==0)
        cum_reward(move_agent)=tot_reward;
        
        % Agents turn
        action_occupied=find(state==1) ; %find the occupied action in a state
        action_available=find(state==0); %find the available action in a state
        state_index=bintodec(state)+1  ;  %get the row index of Q matrix for a state
        
        tempQ=Q(state_index,:);
        tempQ(action_occupied)=-1000*ones(1,length(action_occupied));% filled with a infinety small value in the occupied action
        
        [~,maxQi]=max(tempQ) ; %get the argmax Q(s,a)
        if rand()<eps
            % random move
            
            if(length(action_available)==1)
                action=action_available;
            else
                action=randsample(action_available,1);
            end
        else
            %greedy move
            action=maxQi;
        end
        % stores the symmetrical state and action to be updated
        [z,w]=symmetry(state,action);
        Qsupdate(:,move_agent)=z;
        acupdate(:,move_agent)=w;
        % update the access index of the Q table
        for ii=1:length(w)
            Qaccess(z(ii),w(ii))=Qaccess(z(ii),w(ii))+1;
        end
        
        [r,bn]=posreward(state,action);
        % box marked as won by Agent
        for bi=1:length(bn)
            if(bn(bi)==1)
                [xx,yy,w,h] = rect_box(bi);
                 rectangle('position',[xx,yy,w,h],'curvature',[1 1]);
                hold on;
            end
        end
        if (r>0)
            boxcount_agent=boxcount_agent+r;
            %%turn retained
            i=i+1;
         
            %cimage(xb,yb)=1;
            
        else
           
            r=negreward(state,action);
            
        end
        state(action)=1;
        [a,b]=line_to_coordinate(action);
        %[x,y]=ginput(1)
        %[a,b]=oneclickplot(x,y)
        %ln=coordinate_to_line(a,b)
        
        
        line(a,b,'color',[1 1 0],'linewidth',4);
        hold on
        %imagesc(0:2,0:2,cimage)
        %hold on
        tot_reward=tot_reward+r;
        move_agent=move_agent+1;
        
    else
        %% opponent move
        if (opponent_type=='H')
            [x,y]=ginput(1);
            [a,b]=oneclickplot(x,y);
            ln=coordinate_to_line(a,b);
            
            
            % random opponent
        elseif(opponent_type=='R')
            if(length(action_available)==1)
                ln=action_available;
            else
                ln=randsample(action_available,1);
            end
        end
        [r,bn]=posreward(state,ln);
        % box marked as won by opponent 
        for bi=1:length(bn)
            if(bn(bi)==1)
                [xx,yy,w,h] = rect_box(bi);
                rectangle('position',[xx,yy,w,h]);
                hold on;
            end
        end
        if (r>0)
            boxcount_opponent=boxcount_opponent+r;
            i=i+1;
            
            %cimage(xb,yb)=2;
        end
        state(ln)=1;
        [a,b]=line_to_coordinate(ln);
        line(a,b,'color',[1 0 0],'linewidth',4);
        hold on
        %imagesc(0:2,0:2,cimage);
        %hold on
    end
    if (m==12)
   %      episode ends
   if (opponent_type=='H')
   pause(20);
   end
   % result of the game
         if(boxcount_agent>boxcount_opponent)
             win=win+1;
             wine=wine+1;
             if(opponent_type=='H')
             winH=winH+1;
             end
         elseif(boxcount_agent<boxcount_opponent)
             lose=lose+1;
             losee=losee+1;
             if(opponent_type=='H')
             loseH=loseH+1;
             end
         else
             draw=draw+1;
             drawe=drawe+1;
             if(opponent_type=='H')
             drawH=drawH+1;
             end
         end
         
        tot_episode=cum_episode+episode;
        
        %% updating the Q(s,a) for the episode
        for k=1:(move_agent-1)
            for j=1:length(w)
                Q(Qsupdate(j,k),acupdate(j,k))= (((Qaccess(Qsupdate(j,k),acupdate(j,k))-1)*Q(Qsupdate(j,k),acupdate(j,k))+tot_reward-cum_reward(k)))/(Qaccess(Qsupdate(j,k),acupdate(j,k)));
                
            end
        end
        episode=episode+1;
        
        
       
        
        %% reinitialize at the end of an episode
        clf('reset')
        axis([0,2,0,2]);
        %colormap gray
        %cimage=[0 0;0 0];
        set(gca,'XTick',0:2,'YTick',0:2);
        %set(gca,'LineWidth',4)
        
        grid on
        hold on
        %imagesc(0:2,0:2,cimage)
        i=0;
        m=0;
        state=zeros(1,12);
        Qsupdate=zeros(5,line_no); 
        acupdate=zeros(5,line_no);
        cum_reward=zeros(1,line_no); 
        move_agent=1; 
        tot_reward=0; 
        boxcount_agent=0;
        boxcount_opponent=0;
      
       
      
    end
    
end
cum_episode=cum_episode+episode-1;
if(opponent_type=='H')
    cum_episode_H=cum_episode_H+episode-1;
    count=count+1;
    stats_x(count)=cum_episode_H;
    stats_y(1,count)=winH/(winH+loseH+drawH);
    stats_y(2,count)=drawH/(winH+loseH+drawH);
    stats_y(3,count)=loseH/(winH+loseH+drawH);
end
save('sym_Qval.mat','Q','Qaccess','stats_x','stats_y');
save('sym_Qvalue.mat','cum_episode','count','cum_episode_H','win','lose','draw');
Q;
ind=find(Qaccess~=0);
tQ=Qaccess(ind);
sz=size(ind)
wine
losee
drawe
winp=wine/(wine+losee+drawe);
win;
lose;
draw;
cum_episode
