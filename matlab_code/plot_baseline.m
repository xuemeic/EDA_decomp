clear xlabel xticks;

input.n = 370;
input.X_sparsity = 10;
input.X_supp = 'unif';
input.delta = 0.1; %0.4
input.gamma = 1; %0.4
input.epsilon = 0.2; %0.1
input.K = 1;
input.alpha = 1;

input.tau1 = 2;
input.tau2 = 0.75;
input.H_mode = 'tril';

cutoff = input.n; % graphing purpose

rng(3)
input.B_mode = 'mild';
oo = create_synthetic_data(input);
B1 = oo.B(1:cutoff);

%norm(diff(B1),'inf')*input.n

input.B_mode = 'jump';
input.DB_sparsity = 3;
oo = create_synthetic_data(input);
B2 = oo.B(1:cutoff);


figure(1)
subplot(1,2,1);
plot(linspace(1,cutoff/4,cutoff),B1, LineWidth=1.5)
t1 = title('Baseline model 1: slow varying');
t1.FontSize = 16;
ylim([2,4])
xlabel('Time (seconds)')
xticks(0:15:90)
xlim([0,90])

subplot(1,2,2)
plot(linspace(1,cutoff/4,cutoff),B2, LineWidth=1.5)
ylim([2,4])
t2 = title('Baseline model 2: with jumps');
t2.FontSize = 16;
xlabel('Time (seconds)')
xticks(0:15:90)
xlim([0,90])

%% 
%print("-f1", 'baseline_model', '-djpeg', '-r300')
