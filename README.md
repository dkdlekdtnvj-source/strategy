내가 지금까지 작업한 몇가지 스트래티지를 던져줄게, 이걸 합쳐서 최고의 전략 만들어봐. 가능하면 웹서치 병행하고. 한국어로 대답해줘

1. 1번스크립트 스퀴즈 모멘텀 디럭스
//@version=5
// 매직1분VN (최종 완성본)
//  - 1분 차트 대응을 위해 기본 파라미터 재조정
//  - 방향성 플럭스 연계 조건의 부호 오류 수정으로 신호 정합성 향상
//  - 진입 수량 계산 방식을 명확화하여 레버리지/고정 수량 설정을 안정화
//  - 동적 임계값 사용 시 안전한 폴백 처리 및 필터 기본값을 단타 중심으로 초기화
//  - KCAS 가드·세션 상태를 구조 게이트와 함께 묶어 재진입 중복과 컴파일 충돌(HTF 입력 중복)을 제거

strategy(
     title               = "매직1분VN (최종 완성본)",
     overlay             = true,
     pyramiding          = 0,
     initial_capital     = 500,
     default_qty_type    = strategy.fixed,
     default_qty_value   = 1,
     commission_type     = strategy.commission.percent,
     commission_value    = 0.05
     )

// === Colour Palette ===
const color colup = #ffcfa6
const color coldn = #419fec
const color colpf = #ffd0a6
const color coldf = #4683b4
const color colps = #169b5d
const color colng = #970529
const color colpo = #11cf77
const color colno = #d11645
const color colsh = #ff1100
const color colsm = #ff5e00
const color colsl = #ffa600
const color colnt = #787b8635

// =================================================================================
// === 설정 (Inputs) ==============================================================
// =================================================================================

// === 1. 핵심 지표 설정 ============================================================
gOsc  = "1. 핵심 지표: 스퀴즈 모멘텀"
smb   = input.bool(true , title="모멘텀 히스토그램 표시"     , group=gOsc, inline="osc1")
len   = input.int (12   , title="Oscillator Length"           , group=gOsc, inline="osc1", minval=7 , maxval=50)
sig   = input.int (3    , title="Signal Length"               , group=gOsc, inline="osc2", minval=2 , maxval=7)
useSameLen   = input.bool(false, title="오실레이터 Length를 BB/KC에도 사용"   , group=gOsc, inline="sqzT")
_bbLenIn     = input.int (20  , title="BB Length"                          , group=gOsc, inline="sqzB", minval=5 , maxval=200)
_kcLenIn     = input.int (18  , title="KC Length (ATR)"                    , group=gOsc, inline="sqzK", minval=5 , maxval=200)
sqz_bbLen    = useSameLen ? len : _bbLenIn
sqz_kcLen    = useSameLen ? len : _kcLenIn
sqz_bbMult   = input.float(1.4, title="BB Multiplier"                      , group=gOsc, inline="sqzM", minval=0.1, maxval=10.0, step=0.1)
sqz_kcMult   = input.float(1.0, title="KC Multiplier"                      , group=gOsc, inline="sqzM", minval=0.1, maxval=10.0, step=0.1)
cup   = input.color(colup , title="상승 모멘텀 색상"        , group=gOsc, inline="col1")
cdn   = input.color(coldn , title="하락 모멘텀 색상"      , group=gOsc, inline="col1")
cpf   = input.color(colpf , title="상승 모멘텀 채우기"    , group=gOsc, inline="col2")
cdf   = input.color(coldf , title="하락 모멘텀 채우기"    , group=gOsc, inline="col2")


gDF   = "1. 핵심 지표: 방향성 플럭스"
dfb   = input.bool(true , title="방향성 플럭스 표시"        , group=gDF)
dfl   = input.int (14   , title="Flux Length"                 , group=gDF, minval=7, maxval=50)
dfSmoothLen = input.int(1, title="플럭스 스무딩 Length", group=gDF, minval=1, maxval=100)
dfh   = input.bool(true, title="플럭스 계산에 하이킨아시 캔들 사용"    , group=gDF)
cps   = input.color(colps, title="상승 플럭스 색상"        , group=gDF, inline="dfc1")
cng   = input.color(colng, title="하락 플럭스 색상"        , group=gDF, inline="dfc1")
cpo   = input.color(colpo, title="과매수 플럭스 색상"       , group=gDF, inline="dfc2")
cno   = input.color(colno, title="과매도 플럭스 색상"       , group=gDF, inline="dfc2")

// === 2. 전략 기본 설정 ==========================================================
gStr  = "2. 전략 기본 설정"
startDate     = input.time(timestamp("2025-07-01T00:00:00"), title="백테스트 시작 날짜", group=gStr)
leverage      = input.float(10.0, title="레버리지"      , group=gStr, minval=1.0)
useFixedQty    = input.bool(false, title="고정 계약 수량 사용", group=gStr)
fixedQty       = input.float(1.0, title="고정 계약 수량 (계약)", group=gStr, minval=0.0)
allowLongEntry  = input.bool(true,  title="롱 진입 허용",  group=gStr, inline="pref")
allowShortEntry = input.bool(true,  title="숏 진입 허용", group=gStr, inline="pref")
reentryBars   = input.int (0    , title="재진입 쿨다운 (봉 개수)", group=gStr, minval=0, maxval=100)

gTime = "2.1 기본 모듈: 시간 & 세션"
usePrimarySession = input.bool(false, title="기본 세션 필터 사용", group=gTime)
primarySession   = input.session("0830-0200", title="기본 세션 (거래소 로컬)", group=gTime)
useKstSession    = input.bool(false, title="한국시간 세션 필터", group=gTime)
kstSession       = input.session("0930-0200", title="한국시간 세션", group=gTime)
useDayFilter     = input.bool(false, title="요일 필터 사용", group=gTime)
monOk = input.bool(true , title="월", group=gTime, inline="dow1")
tueOk = input.bool(true , title="화", group=gTime, inline="dow1")
wedOk = input.bool(true , title="수", group=gTime, inline="dow1")
thuOk = input.bool(true , title="목", group=gTime, inline="dow1")
friOk = input.bool(true , title="금", group=gTime, inline="dow2")
satOk = input.bool(false, title="토", group=gTime, inline="dow2")
sunOk = input.bool(false, title="일", group=gTime, inline="dow2")

gGuard = "2.2 기본 모듈: 거래 가드"
useDailyLossGuard   = input.bool(false, title="일일 손실 한도 사용", group=gGuard)
dailyLossLimit      = input.float(80.0, title="일일 손실 한도 ($)", group=gGuard, minval=0.0, step=1.0)
useDailyProfitLock  = input.bool(false, title="일일 이익 잠금", group=gGuard)
dailyProfitTarget   = input.float(120.0, title="일일 이익 목표 ($)", group=gGuard, minval=0.0, step=1.0)
useWeeklyProfitLock = input.bool(false, title="주간 이익 잠금", group=gGuard)
weeklyProfitTarget  = input.float(250.0, title="주간 이익 목표 ($)", group=gGuard, minval=0.0, step=1.0)
useLossStreakGuard  = input.bool(false, title="연속 손실 중지", group=gGuard)
maxConsecutiveLosses = input.int(3, title="허용 연속 손실", group=gGuard, minval=1, maxval=10)
useCapitalGuard     = input.bool(false, title="자본 드로우다운 가드", group=gGuard)
capitalGuardPct     = input.float(20.0, title="자본 드로우다운 한도 (%)", group=gGuard, minval=1.0, maxval=100.0, step=1.0)
maxDailyLosses      = input.int(0, title="일일 손실 거래 수 제한", group=gGuard, minval=0, maxval=10)
maxWeeklyDD         = input.float(0.0, title="주간 드로우다운 한도 (%)", group=gGuard, minval=0.0, maxval=50.0, step=0.1)
maxGuardFires       = input.int(0, title="가드 강제 청산 허용 횟수", group=gGuard, minval=0, maxval=20)
useGuardExit        = input.bool(false, title="청산가 선제 가드 사용", group=gGuard)
maintenanceMarginPct = input.float(0.5, title="유지 증거금 %", group=gGuard, minval=0.1, maxval=5.0, step=0.05)
preemptTicks        = input.int(8, title="선제 청산 틱", group=gGuard, minval=0, maxval=50)

gRisk = "2.3 KCAS 리스크 모듈"
positionSizingMode = input.string("Notional", title="사이징 모드", options=["Risk-Based", "Notional"], group=gRisk)
riskSizingType     = input.string("Fixed Fractional", title="리스크 포지션 타입", options=["Fixed Fractional", "Fixed Lot"], group=gRisk)
baseRiskPct        = input.float(0.6, title="기본 리스크 %", group=gRisk, minval=0.1, step=0.05)
fixedContractSize  = input.float(1.0, title="고정 계약 수량", group=gRisk, minval=0.001, step=0.1)
notionalSizingType = input.string("Equity %", title="노션널 기준", options=["Fixed USD", "Equity %"], group=gRisk)
notionalSizingValue = input.float(20.0, title="노션널 값 (USD 또는 %)", group=gRisk, minval=1.0)
slipTicks          = input.int(1, title="슬리피지 (틱)", group=gRisk, minval=0, maxval=50)
useWallet          = input.bool(false, title="월렛 시스템 사용", group=gRisk)
profitReservePct   = input.float(20.0, title="수익 적립 비율 %", group=gRisk, minval=0.0, maxval=100.0, step=1.0) / 100.0
applyReserveToSizing = input.bool(true, title="적립금 제외 후 사이징", group=gRisk)
minTradableCapital = input.float(250.0, title="최소 거래 가능 자본 ($)", group=gRisk, minval=50.0)
useDrawdownScaling = input.bool(false, title="드로우다운 리스크 축소", group=gRisk)
drawdownTriggerPct = input.float(7.0, title="드로우다운 트리거 %", group=gRisk, minval=1.0, maxval=50.0)
drawdownRiskScale  = input.float(0.5, title="드로우다운 리스크 배율", group=gRisk, minval=0.1, maxval=1.0, step=0.05)
usePerfAdaptiveRisk = input.bool(false, title="성과 적응 리스크 (PAR)", group=gRisk)
parLookback        = input.int(6, title="PAR 거래 수 집계", group=gRisk, minval=2, maxval=20)
parMinTrades       = input.int(3, title="PAR 최소 거래 수", group=gRisk, minval=1, maxval=20)
parHotWinRate      = input.float(65.0, title="핫스트릭 승률 %", group=gRisk, minval=40.0, maxval=90.0, step=0.5)
parColdWinRate     = input.float(35.0, title="콜드스트릭 승률 %", group=gRisk, minval=5.0, maxval=60.0, step=0.5)
parHotRiskMult     = input.float(1.25, title="핫스트릭 리스크 배율", group=gRisk, minval=1.0, maxval=2.0, step=0.05)
parColdRiskMult    = input.float(0.35, title="콜드스트릭 리스크 배율", group=gRisk, minval=0.0, maxval=1.0, step=0.05)
parPauseOnCold     = input.bool(true, title="콜드스트릭 시 진입 중지", group=gRisk)
useVolatilityGuard = input.bool(false, title="ATR 변동성 가드", group=gRisk)
volatilityLookback = input.int(50, title="ATR %% 기간", group=gRisk, minval=10, maxval=200)
volatilityLowerPct = input.float(0.15, title="ATR %% 하한", group=gRisk, minval=0.05, step=0.05)
volatilityUpperPct = input.float(2.5, title="ATR %% 상한", group=gRisk, minval=0.2, step=0.05)

// === 3. 진입 조건 설정 ===========================================================
gSig   = "3. 진입 조건: 모멘텀 임계값"
useSymThreshold    = input.bool(false , title="고정 임계값 대칭 사용" , group=gSig, inline="thType")
useDynamicThresh   = input.bool(true , title="동적 임계값 사용"  , group=gSig, inline="thType")
statThreshold      = input.float(38.0, title="고정 임계값 (절대값)"         , group=gSig, inline="stat", minval=0 , maxval=200 , step=0.5)
buyThreshold       = input.float(36.0, title="매수 임계값 (절대값)"            , group=gSig, inline="sep" , minval=0 , maxval=200 , step=0.5)
sellThreshold      = input.float(36.0, title="매도 임계값 (절대값)"           , group=gSig, inline="sep" , minval=0 , maxval=200 , step=0.5)
dynLen             = input.int (21   , title="동적 임계값 Length"       , group=gSig, inline="dyn" , minval=5 , maxval=300)
dynMult            = input.float(1.1 , title="동적 임계값 Multiplier"   , group=gSig, inline="dyn" , minval=0.1, maxval=5.0, step=0.1)

// === 4. 청산 조건 설정 ==========================================================
gExit = "4. 청산 조건"
exitOpposite      = input.bool(true, title="반대 신호에 청산", group=gExit)
useMomFade        = input.bool(false, title="모멘텀 페이드 엑싯 사용" , group=gExit)
momFadeBars       = input.int (1  , title="페이드 감소 확인 (봉 수)"     , group=gExit, minval=1, maxval=10)
momFadeLen        = input.int (20 , title="모멘텀 페이드 Length", group=gExit, minval=5, maxval=200)
momFadeBbMult     = input.float(2.0, title="페이드 BB 배수", group=gExit, inline="mfmult", minval=0.1, maxval=5.0, step=0.1)
momFadeKcMult     = input.float(1.5, title="페이드 KC 배수", group=gExit, inline="mfmult", minval=0.1, maxval=5.0, step=0.1)
momFadeUseTrueRange = input.bool(true, title="TR 기반 KC 사용 (페이드)", group=gExit)
momFadeZeroDelay  = input.int (0  , title="제로 크로스 대기", group=gExit, inline="mfopt", minval=0, maxval=20)
momFadeMinAbs     = input.float(0.0, title="최소 절댓값", group=gExit, inline="mfopt", minval=0.0, maxval=100.0, step=0.1)
minHoldBars       = input.int (0  , title="최소 포지션 유지 (봉 개수)"    , group=gExit, minval=0, maxval=100)
useStopLoss       = input.bool(false, title="고정 전고/전저 손절 사용"        , group=gExit)
stopLookback      = input.int (5  , title="손절 탐색 범위 (봉 개수)"   , group=gExit, minval=1, maxval=100)
useAtrTrail       = input.bool(false, title="ATR 트레일링 스탑 사용", group=gExit)
atrTrailLen       = input.int (7 , title="ATR 트레일링 Length"       , group=gExit, minval=1, maxval=100)
atrTrailMult      = input.float(2.5, title="ATR 트레일링 Multiplier"   , group=gExit, minval=0.1, maxval=10.0, step=0.1)
useBreakevenStop  = input.bool(false, title="본절 로스 사용", group=gExit)
breakevenMult     = input.float(1.0, title="본절 로스 발동 ATR Multiplier", group=gExit, minval=0.1, maxval=10.0, step=0.1)
usePivotStop      = input.bool(false, title="피봇 기반 손절 사용"  , group=gExit)
pivotLen          = input.int (5  , title="Pivot Length"           , group=gExit, minval=2, maxval=50)
usePivotHtf       = input.bool(false, title="피봇 계산에 상위 타임프레임 사용", group=gExit)
pivotTf           = input.timeframe("5", title="피봇 상위 타임프레임", group=gExit)
useAtrProfit      = input.bool(false, title="ATR 익절 사용" , group=gExit)
atrProfitMult     = input.float(2.0, title="ATR 익절 Multiplier"  , group=gExit, minval=0.1, maxval=10.0, step=0.1)
useDynVol         = input.bool(false, title="동적 변동성 적용 (손익절 거리 조절)", group=gExit)
useStopDistanceGuard = input.bool(false, title="손절 거리 가드", group=gExit)
maxStopAtrMult    = input.float(2.8, title="최대 손절 거리 (ATR 배수)", group=gExit, minval=0.5, maxval=5.0, step=0.1)
useTimeStop       = input.bool(false, title="시간 기반 청산", group=gExit)
maxHoldBars       = input.int (45 , title="최대 보유 봉수", group=gExit, minval=5, maxval=2000)
useKASA           = input.bool(false, title="KASA 조기 익절", group=gExit)
kasa_rsiLen       = input.int (14 , title="KASA RSI Length", group=gExit, minval=1)
kasa_rsiOB        = input.float(72.0, title="RSI 과매수", group=gExit, minval=50.0, maxval=100.0, step=0.5)
kasa_rsiOS        = input.float(28.0, title="RSI 과매도", group=gExit, minval=0.0, maxval=50.0, step=0.5)
useBETiers        = input.bool(false, title="Break-even 계층", group=gExit)

gShock = "4. 청산 조건: 변동성 쇼크 방어"
useShock    = input.bool(false, "변동성 쇼크 방어 사용", group=gShock)
atrFastLen  = input.int(5, "ATR Fast", group=gShock)
atrSlowLen  = input.int(20,"ATR Slow SMA of Fast", group=gShock)
shockMult   = input.float(2.5, "쇼크 감지 ATR Multiplier", step=0.1, group=gShock)
shockAction = input.string("손절 타이트닝", "쇼크 발생 시 액션", options=["즉시 청산","손절 타이트닝"], group=gShock)

// === 5. 추가 진입 필터 =========================================================
gFilt = "5. 추가 진입 필터"
useAdx       = input.bool(false , title="ADX 필터 사용"        , group=gFilt)
adxLen       = input.int (10   , title="ADX Length"            , group=gFilt, minval=5, maxval=100)
adxThresh    = input.float(15.0, title="ADX 임계값"         , group=gFilt, minval=5.0, maxval=100.0)

useEma       = input.bool(false, title="EMA 필터 사용"        , group=gFilt)
emaFastLen   = input.int (8    , title="Fast EMA Length"        , group=gFilt, minval=1, maxval=200)
emaSlowLen   = input.int (20   , title="Slow EMA Length"        , group=gFilt, minval=1, maxval=400)
emaMode      = input.string("Trend", title="EMA 필터 모드", options=["Crossover","Trend"], group=gFilt)

useBb        = input.bool(false, title="볼린저밴드 필터 사용"  , group=gFilt)
bbLenFilter  = input.int (20   , title="BB Filter Length"       , group=gFilt, minval=5, maxval=100)
bbMultFilter = input.float(2.0 , title="BB Filter Mult"         , group=gFilt, minval=0.5, maxval=5.0)

useStochRsi  = input.bool(false, title="StochRSI 필터 사용"   , group=gFilt)
stochLen     = input.int (14   , title="StochRSI Length"        , group=gFilt, minval=5, maxval=50)
stochOB      = input.float(80.0, title="StochRSI 과매수"    , group=gFilt, minval=50.0, maxval=100.0)
stochOS      = input.float(20.0, title="StochRSI 과매도"      , group=gFilt, minval=0.0, maxval=50.0)

useObv       = input.bool(false, title="OBV 기울기 필터 사용"   , group=gFilt)
obvSmoothLen = input.int (3    , title="OBV EMA Length"         , group=gFilt, minval=1, maxval=50)

useAtrDiff   = input.bool(false, title="ATR 차이 필터 사용", group=gFilt)
adxAtrTf     = input.timeframe("5", "ADX/ATR 필터용 상위 타임프레임", group=gFilt)

useHtfTrend  = input.bool(false, title="상위 타임프레임 추세 필터 사용", group=gFilt)
htfTrendTf   = input.timeframe("240", "상위 타임프레임"          , group=gFilt)
htfMaLen     = input.int (20   , title="상위 타임프레임 MA Length"     , group=gFilt, minval=1, maxval=200)


gAdd = "5. 추가 진입 필터: 기타"
useHmaFilter  = input.bool(false, title="HMA 트렌드 필터 사용", group=gAdd)
hmaLen        = input.int(20, title="HMA Length", group=gAdd, minval=1, maxval=200)

useRangeFilter = input.bool(false, title="상위봉 레인지 필터 사용", group=gAdd)
rangeTf        = input.timeframe("5", title="레인지 측정 상위 시간", group=gAdd)
rangeBars      = input.int(20, title="레인지 측정 봉 수", group=gAdd, minval=5, maxval=100)
rangePercent   = input.float(1.0, title="레인지 기준 퍼센트 (%)", group=gAdd, minval=0.1, maxval=10.0, step=0.1)

useSessionFilter = input.bool(false, title="미국장 세션만 거래", group=gAdd)
usSession        = input.session("0930-0000", title="미국장 세션 (현지시간)", group=gAdd)

gCtxKcas = "5. 추가 진입 필터: KCAS 컨텍스트"
useRegimeFilter = input.bool(false, title="상위봉 레짐 필터", group=gCtxKcas)
ctxHtfTf        = input.timeframe("240", title="상위봉 타임프레임", group=gCtxKcas)
ctxHtfEmaLen    = input.int(120, title="상위봉 EMA 길이", group=gCtxKcas, minval=20, maxval=400)
ctxHtfAdxLen    = input.int(14, title="상위봉 ADX 길이", group=gCtxKcas, minval=5, maxval=50)
ctxHtfAdxTh     = input.float(22.0, title="상위봉 ADX 임계", group=gCtxKcas, minval=5.0, maxval=50.0, step=0.5)
useMicroTrend   = input.bool(false, title="EMA 클라우드 필터", group=gCtxKcas)
emaFastLenBase  = input.int(21, title="EMA 빠른선", group=gCtxKcas, minval=5, maxval=100)
emaSlowLenBase  = input.int(55, title="EMA 느린선", group=gCtxKcas, minval=10, maxval=200)
useTrendBias    = input.bool(false, title="추세 EMA 필터", group=gCtxKcas)
trendLenBase    = input.int(200, title="추세 EMA 길이", group=gCtxKcas, minval=20, maxval=400)
useConfBias     = input.bool(false, title="확인 EMA 필터", group=gCtxKcas)
confLenBase     = input.int(55, title="확인 EMA 길이", group=gCtxKcas, minval=10, maxval=300)
useSlopeFilter  = input.bool(false, title="EMA 기울기 필터", group=gCtxKcas)
slopeLookback   = input.int(8, title="기울기 룩백", group=gCtxKcas, minval=1, maxval=50)
slopeMinPct     = input.float(0.06, title="최소 기울기 (%)", group=gCtxKcas, minval=0.0, maxval=1.0, step=0.01)
useDistanceGuard = input.bool(false, title="가격 이격 가드", group=gCtxKcas)
distanceAtrLen   = input.int(21, title="이격 ATR 길이", group=gCtxKcas, minval=5, maxval=200)
distanceMaxAtr   = input.float(2.4, title="최대 이격 (ATR)", group=gCtxKcas, minval=0.5, maxval=5.0, step=0.1)
useEquitySlopeFilter = input.bool(false, title="순자산 기울기 필터", group=gCtxKcas)
eqSlopeLen       = input.int(120, title="순자산 기울기 길이", group=gCtxKcas, minval=20, maxval=500)

// === 6. 부가 기능 및 시각화 ======================================================
gRev = "6. 부가 기능"
useReversal       = input.bool(false, title="청산 후 자동 반대매매 진입", group=gRev)
reversalDelaySec  = input.float(0.0,  title="반대매매 지연 시간 (초)", group=gRev, minval=0.0, maxval=3600.0, step=1.0)

gDiv   = "6. 부가 기능: 다이버전스"
trs    = input.int (25 , title="다이버전스 민감도"      , group=gDiv, minval=20, maxval=40)
dbl    = input.bool(true, title="다이버전스 선 표시"       , group=gDiv)
dbs    = input.bool(true, title="다이버전스 라벨 표시"      , group=gDiv)
cdu    = input.color(colpo, title="상승 다이버전스 색상" , group=gDiv, inline="divCol")
cdd    = input.color(colno, title="하락 다이버전스 색상" , group=gDiv, inline="divCol")

gShow  = "6. 부가 기능: 게이지"
gds    = input.string("Both", title="게이지 표시"           , options=["Both","Bull","Bear","None"], group=gShow)
cgp    = input.color(colps, title="상승 게이지 색상"         , group=gShow, inline="gCol")
cgn    = input.color(colng, title="하락 게이지 색상"         , group=gShow, inline="gCol")

gHud = "6. 부가 기능: HUD"
showHudPanel   = input.bool(true, title="KCAS HUD 표시", group=gHud)
hudPosition    = input.string("Top Right", title="HUD 위치", options=["Top Left","Top Right","Bottom Left","Bottom Right"], group=gHud)
showDebugPanel = input.bool(false, title="디버그 패널 표시", group=gHud)

// === 7. KASIA vNext 모듈 =======================================================
group_kasia = "7. KASIA vNext 모듈"
useSqzGate   = input.bool(false, "스퀴즈 릴리즈 게이트 사용", group=group_kasia)
bbLen_vn     = input.int(20, "BB Length", minval=5, maxval=200, group=group_kasia)
bbMult_vn    = input.float(1.5, "BB Mult", minval=0.1, maxval=10, step=0.1, group=group_kasia)
kcLen_vn     = input.int(14, "KC (ATR) Length", minval=5, maxval=200, group=group_kasia)
kcMult_vn    = input.float(1.0, "KC Mult", minval=0.1, maxval=10, step=0.1, group=group_kasia)
releaseBars  = input.int(5, "릴리즈 후 유효 봉 수", minval=1, maxval=50, group=group_kasia)

useBOS    = input.bool(false, "BOS 필요", group=group_kasia)
useCHOCH  = input.bool(false, "CHoCH 필요", group=group_kasia)
choch_stateBars = input.int(5, title="CHoCH 신호 유효 봉 수", group=group_kasia, minval=1)
structureGateMode = input.string("Any", "구조 돌파 조건", options=["Any","All"], group=group_kasia)
bosTf     = input.timeframe("15", "구조 분석 타임프레임", group=group_kasia)
pivotLeft_vn    = input.int(5, "구조분석 피봇 왼쪽 강도", minval=1, maxval=20, group=group_kasia)
pivotRight_vn   = input.int(5, "구조분석 피봇 오른쪽 강도", minval=1, maxval=20, group=group_kasia)

// === 8. 얼럿 메시지 ===========================================================
gMsg = "8. 얼럿 메시지"
alertLongEntry  = input.string('{"action":"enter_long"}', title="롱 진입", group=gMsg)
alertShortEntry = input.string('{"action":"enter_short"}', title="숏 진입", group=gMsg)
alertExitLong   = input.string('{"action":"exit_long"}' , title="롱 청산", group=gMsg)
alertExitShort  = input.string('{"action":"exit_short"}', title="숏 청산", group=gMsg)
alertPartialLong  = input.string('{"action":"partial_exit_long"}',  title="롱 부분청산",  group=gMsg)
alertPartialShort = input.string('{"action":"partial_exit_short"}', title="숏 부분청산", group=gMsg)


// =================================================================================
// === 계산 (Calculations) ========================================================
// =================================================================================

// --- 비활성화된 기능들 ---
useEventFilter = false
useStochRsiExit = false
usePartialProfit = false
usePeakTrail = false
useRiskLimitStop = false
useVSpike = false
useScore_vn = false

// --- KCAS 기본 모듈: 시간 & 리스크 가드 계산 ---
bool sessionAllowed = not usePrimarySession or not na(time(timeframe.period, primarySession))
bool kstAllowed    = not useKstSession or not na(time(timeframe.period, kstSession, "Asia/Seoul"))
bool dayAllowed    = not useDayFilter or ((dayofweek == dayofweek.monday and monOk) or (dayofweek == dayofweek.tuesday and tueOk) or (dayofweek == dayofweek.wednesday and wedOk) or (dayofweek == dayofweek.thursday and thuOk) or (dayofweek == dayofweek.friday and friOk) or (dayofweek == dayofweek.saturday and satOk) or (dayofweek == dayofweek.sunday and sunOk))
bool isBacktestWindow = time >= startDate

var float tickSize = syminfo.mintick
var float tradableCapital = strategy.initial_capital
var float withdrawable = 0.0
var float peakEquity = strategy.initial_capital
float slipBuffer = tickSize * slipTicks

if barstate.isconfirmed
    float newProfit = strategy.netprofit - nz(strategy.netprofit[1])
    if useWallet and newProfit > 0
        withdrawable += newProfit * profitReservePct
    float effectiveEquity = useWallet and applyReserveToSizing ? strategy.equity - withdrawable : strategy.equity
    tradableCapital := math.max(effectiveEquity, strategy.initial_capital * 0.01)
    peakEquity := math.max(peakEquity, strategy.equity)

float currentDD = peakEquity > 0 ? (peakEquity - strategy.equity) / peakEquity * 100.0 : 0.0
float scaledRiskPct = useDrawdownScaling and currentDD > drawdownTriggerPct ? baseRiskPct * drawdownRiskScale : baseRiskPct

var float[] recentTradeResults = array.new_float()
var int lastClosedCount = 0
int closedCount = strategy.closedtrades
if usePerfAdaptiveRisk and closedCount > lastClosedCount
    for idx = lastClosedCount to closedCount - 1
        float tradeProfit = strategy.closedtrades.profit(idx)
        array.push(recentTradeResults, tradeProfit)
        if array.size(recentTradeResults) > parLookback
            array.shift(recentTradeResults)
    lastClosedCount := closedCount
else if not usePerfAdaptiveRisk
    lastClosedCount := closedCount

int recentTrades = array.size(recentTradeResults)
int recentWins = 0
int recentLosses = 0
if usePerfAdaptiveRisk and recentTrades > 0
    for i = 0 to recentTrades - 1
        float plTrade = array.get(recentTradeResults, i)
        recentWins   += plTrade > 0 ? 1 : 0
        recentLosses += plTrade < 0 ? 1 : 0

float recentWinRate = usePerfAdaptiveRisk and recentTrades > 0 ? recentWins / recentTrades * 100.0 : na
bool isHotStreak = usePerfAdaptiveRisk and not na(recentWinRate) and recentTrades >= parMinTrades and recentWinRate >= parHotWinRate
bool isColdStreak = usePerfAdaptiveRisk and not na(recentWinRate) and recentTrades >= parMinTrades and recentWinRate <= parColdWinRate
float perfRiskMult = usePerfAdaptiveRisk ? (isHotStreak ? parHotRiskMult : isColdStreak ? parColdRiskMult : 1.0) : 1.0
float finalRiskPct = scaledRiskPct * perfRiskMult
string parStateLabel = not usePerfAdaptiveRisk ? "OFF" : isHotStreak ? "HOT" : isColdStreak ? "COLD" : "NEUTRAL"
string parWinLabel = na(recentWinRate) ? "-" : str.tostring(recentWinRate, "##.##") + "%"

var float dailyStartCapital = tradableCapital
var float dailyPeakCapital = tradableCapital
var float weekStartEquity = strategy.initial_capital
var float weekPeakEquity = strategy.initial_capital
var int dailyLosses = 0
var int lossStreak = 0
var bool guardFrozen = false
var int guardFiredTotal = 0

bool newDay = ta.change(dayofmonth) != 0
if newDay
    dailyStartCapital := tradableCapital
    dailyPeakCapital := tradableCapital
    dailyLosses := 0
    guardFrozen := false

bool newWeek = ta.change(weekofyear) != 0
if newWeek
    weekStartEquity := strategy.equity
    weekPeakEquity := strategy.equity
else
    weekPeakEquity := math.max(weekPeakEquity, strategy.equity)

dailyPeakCapital := math.max(dailyPeakCapital, tradableCapital)
float dailyPnl = tradableCapital - dailyStartCapital
float weeklyPnl = strategy.equity - weekStartEquity
float weeklyDD = weekPeakEquity > 0 ? (weekPeakEquity - strategy.equity) / weekPeakEquity * 100.0 : 0.0

bool dailyLossBreached = useDailyLossGuard and dailyPnl <= -math.abs(dailyLossLimit)
bool dailyProfitReached = useDailyProfitLock and dailyPnl >= math.abs(dailyProfitTarget)
bool weeklyProfitReached = useWeeklyProfitLock and weeklyPnl >= math.abs(weeklyProfitTarget)
bool lossStreakBreached = useLossStreakGuard and lossStreak >= maxConsecutiveLosses
bool capitalBreached    = useCapitalGuard and strategy.equity <= strategy.initial_capital * (1 - capitalGuardPct / 100.0)
bool weeklyDDBreached   = maxWeeklyDD > 0 and weeklyDD >= maxWeeklyDD

if strategy.losstrades > strategy.losstrades[1]
    dailyLosses += 1
    lossStreak += 1
if strategy.wintrades > strategy.wintrades[1]
    lossStreak := 0

float atrPct = close != 0 ? ta.atr(volatilityLookback) / close * 100.0 : 0.0
bool isVolatilityOK = not useVolatilityGuard or (atrPct >= volatilityLowerPct and atrPct <= volatilityUpperPct)

bool stopByCapital = tradableCapital < minTradableCapital
bool stopByPerf = usePerfAdaptiveRisk and parPauseOnCold and isColdStreak
bool lossCountBreached = maxDailyLosses > 0 and dailyLosses >= maxDailyLosses
bool guardFireLimit = maxGuardFires > 0 and guardFiredTotal >= maxGuardFires

bool guardFrozenPrev = guardFrozen
bool shouldFreeze = dailyLossBreached or dailyProfitReached or weeklyProfitReached or lossStreakBreached or capitalBreached or stopByCapital or stopByPerf or lossCountBreached or weeklyDDBreached or guardFireLimit
if shouldFreeze
    guardFrozen := true

bool guardActivated = guardFrozen and not guardFrozenPrev

bool haltReasons = guardFrozen
bool canTrade = isBacktestWindow and sessionAllowed and kstAllowed and dayAllowed and not haltReasons and isVolatilityOK

bool guardClosedThisBar = false
if guardActivated and strategy.position_size != 0
    strategy.close_all(comment="Guard Halt")
    guardFiredTotal += 1
    guardClosedThisBar := true

kasia_guard_price(entryPrice, direction, qty) =>
    if qty == 0.0
        entryPrice
    else
        float initialMargin = (qty * entryPrice) / leverage
        float maintMargin   = (qty * entryPrice) * (maintenanceMarginPct / 100.0)
        float offset        = (initialMargin - maintMargin) / qty
        direction == 1 ? entryPrice - offset : entryPrice + offset

if useGuardExit and strategy.position_size != 0 and not guardClosedThisBar
    float guardEntry = strategy.position_avg_price
    int guardDir = strategy.position_size > 0 ? 1 : -1
    float guardQty = math.abs(strategy.position_size)
    float liqPrice = kasia_guard_price(guardEntry, guardDir, guardQty)
    float preemptPrice = guardDir == 1 ? liqPrice + preemptTicks * tickSize : liqPrice - preemptTicks * tickSize
    bool hitGuard = guardDir == 1 ? low <= preemptPrice : high >= preemptPrice
    if hitGuard
        strategy.close(guardDir == 1 ? "Long" : "Short", comment="Guard Exit")
        guardFrozen := true
        guardFiredTotal += 1
        guardClosedThisBar := true

// --- 포지션 수량 계산 ---
calcRiskQty(stopDistance, signalRiskMult) =>
    if positionSizingMode == "Risk-Based" and not na(stopDistance) and stopDistance > 0
        if riskSizingType == "Fixed Lot"
            fixedContractSize * math.max(signalRiskMult, 0.0)
        else
            float riskPct = math.max(finalRiskPct * signalRiskMult, 0.0)
            float riskCapital = tradableCapital * riskPct / 100.0
            riskCapital > 0 ? riskCapital / (stopDistance + slipBuffer) : na
    else
        na

calcNotionalQty(closePrice, signalRiskMult) =>
    float riskScale = baseRiskPct > 0 ? finalRiskPct / baseRiskPct : 1.0
    float scaleMult = math.max(signalRiskMult, 0.0) * math.max(riskScale, 0.0)
    if useFixedQty
        math.max(fixedQty * scaleMult, 0.0)
    else
        float baseUsd = notionalSizingType == "Fixed USD" ? notionalSizingValue : tradableCapital * (notionalSizingValue / 100.0)
        float adjUsd = math.max(baseUsd * scaleMult, 0.0)
        closePrice > 0 ? (adjUsd * leverage) / closePrice : 0.0

calcOrderSize(closePrice, stopDistance, signalRiskMult) =>
    float qtyRisk = calcRiskQty(stopDistance, signalRiskMult)
    float qtyNotional = calcNotionalQty(closePrice, signalRiskMult)
    if positionSizingMode == "Risk-Based" and not na(qtyRisk)
        math.max(qtyRisk, 0.0)
    else
        na(qtyNotional) ? 0.0 : math.max(qtyNotional, 0.0)

// --- KASIA vNext 게이트 계산 ---
dev_vn   = ta.stdev(close, bbLen_vn) * bbMult_vn
atrKC_vn = ta.atr(kcLen_vn) * kcMult_vn
sqOn_vn  = dev_vn < atrKC_vn
sqRel_vn = (nz(sqOn_vn[1]) and not sqOn_vn) or ta.crossover(dev_vn, atrKC_vn)
relOk_vn = ta.barssince(sqRel_vn) <= releaseBars

// --- [로직 개선] KASIA 구조 분석 계산 ---
ph_vn = request.security(syminfo.tickerid, bosTf, ta.pivothigh(high, pivotLeft_vn, pivotRight_vn), lookahead=barmerge.lookahead_off)
pl_vn = request.security(syminfo.tickerid, bosTf, ta.pivotlow(low,  pivotLeft_vn, pivotRight_vn), lookahead=barmerge.lookahead_off)
float lastPH_vn = ta.valuewhen(not na(ph_vn), ph_vn, 0)
float lastPL_vn = ta.valuewhen(not na(pl_vn), pl_vn, 0)

// BOS (Break of Structure): 최근 스윙 고점/저점 돌파 지속
bool bosLong_vn  = close > lastPH_vn
bool bosShort_vn = close < lastPL_vn

// CHoCH (Change of Character): 최근 스윙 고점/저점 돌파 발생
bool chochLong_event  = ta.crossover(close, lastPH_vn)
bool chochShort_event = ta.crossunder(close, lastPL_vn)
bool chochLong_state = ta.barssince(chochLong_event) < choch_stateBars
bool chochShort_state = ta.barssince(chochShort_event) < choch_stateBars


// --- 타입 정의 ---
type bar
    float o
    float h
    float l
    float c
    int   i

type osc
    float o
    float s

type squeeze
    bool  h
    bool  m
    bool  l

type gauge
    float u
    float l
    color c
    bool  p

type divergence
    float p
    float s
    int   i

type alerts
    bool b
    bool s
    bool u
    bool d
    bool p
    bool n
    bool x
    bool y
    bool a
    bool c
    bool q
    bool w
    bool h
    bool m
    bool l
    bool e
    bool f

// --- 피봇 추적용 변수 ---
var float pivotHighStop = na
var float pivotLowStop  = na

// --- 메소드 (함수) 정의 ---
method src(bar b, simple string src) =>
    float x = switch src
        'oc2'   => math.avg(b.o, b.c)
        'hl2'   => math.avg(b.h, b.l)
        'hlc3'  => math.avg(b.h, b.l, b.c)
        'ohlc4' => math.avg(b.o, b.h, b.l, b.c)
        'hlcc4' => math.avg(b.h, b.l, b.c, b.c)
    x
method ha(bar b, simple bool p = true) =>
    var bar x = bar.new(na, na, na, na, na)
    x.c := b.src('ohlc4')
    x    := bar.new(na(x.o[1]) ? b.src('oc2') : nz(x.src('oc2')[1]), math.max(b.h, math.max(x.o, x.c)), math.min(b.l, math.min(x.o, x.c)), x.c, b.i)
    p ? x : b
method atr(bar b, simple int len = 1) =>
    float tr = na(b.h[1]) ? (b.h - b.l) : math.max(math.max(b.h - b.l, math.abs(b.h - b.c[1])), math.abs(b.l - b.c[1]))
    len == 1 ? tr : ta.rma(tr, len)
method stdev(float src, simple int len) =>
    float sq  = 0., psq = 0., sum = 0.
    for k = 0 to len - 1
        val  = nz(src[k])
        psq := sq
        sq  += (val - sq) / (1 + k)
        sum += (val - sq) * (val - psq)
    math.sqrt(sum / (len - 1))
method osc(bar b, simple int sig, simple int len) =>
    float av = ta.sma(b.src('hl2'), len)
    bar   z  = bar.new(b.o, ta.highest(b.h, len), ta.lowest(b.l, len), b.c, b.i)
    float x  = ta.linreg((z.c - math.avg(z.src('hl2'), av)) / z.atr() * 100, len, 0)
    osc.new(x, ta.sma(x, sig))
method dfo(bar b, simple int len) =>
    float tr = b.atr(len)
    float up = ta.rma(math.max(ta.change(b.h), 0), len) / tr
    float dn = ta.rma(math.max(ta.change(b.l) * -1, 0), len) / tr
    float x  = ta.rma((up - dn) / (up + dn), len / 2) * 100
    osc.new(x, x > +25 ? (x - 25) : x < -25 ? (x + 25) : na)
method sqz(bar b, simple int bbLen, simple int kcLen, simple float bbMult, simple float kcMult) =>
    array<bool> sqzArr = array.new_bool()
    float dev  = b.c.stdev(bbLen) * bbMult
    float atrv = b.atr(kcLen) * kcMult
    for i = 2 to 4
        sqzArr.unshift(dev < (atrv * 0.25 * i))
    squeeze.new(sqzArr.pop(), sqzArr.pop(), sqzArr.pop())
method draw(bar b, osc o, simple int trs, simple bool s) =>
    var divergence d = divergence.new()
    bool u = ta.crossunder(o.o, o.s)
    bool l = ta.crossover(o.o, o.s)
    float x = o.s
    bool p = false
    if o.o > trs and u and barstate.isconfirmed
        if na(d.p)
            d := divergence.new(b.h, x, b.i)
            p := false
        else if b.h > d.p and x < d.s
            if s
                line.new(d.i, d.s, b.i, x, xloc.bar_index, extend.none, cdd)
            d := divergence.new()
            p := true
        else
            d := divergence.new(b.h, x, b.i)
            p := false
    if o.o < -trs and l and barstate.isconfirmed
        if na(d.p)
            d := divergence.new(b.l, x, b.i)
            p := false
        else if b.l < d.p and x > d.s
            if s
                line.new(d.i, d.s, b.i, x, xloc.bar_index, extend.none, cdu)
            d := divergence.new()
            p := true
        else
            d := divergence.new(b.l, x, b.i)
            p := false
    p
getPivots(int len) =>
    float ph = ta.pivothigh(high, len, len)
    float pl = ta.pivotlow(low,  len, len)
    [ph, pl]

// --- 핵심 지표 계산 ---
bar b = bar.new(open, high, low, close, bar_index)
squeeze s = b.sqz(sqz_bbLen, sqz_kcLen, sqz_bbMult, sqz_kcMult)
osc     o = b.osc(sig, len)
osc     v = b.ha(dfh).dfo(dfl)
float v_o_sm = dfSmoothLen > 1 ? ta.sma(v.o, dfSmoothLen) : v.o
float v_s_sm = dfSmoothLen > 1 ? ta.sma(v.s, dfSmoothLen) : v.s
bool p = b.draw(o, trs, dbl)
gauge uG = gauge.new(+75, +70, v_o_sm > 0 and o.o > 0 ? cgp : v_o_sm > 0 or o.o > 0 ? color.new(cgp, 40) : colnt, gds == 'Both' or gds == 'Bull')
gauge dG = gauge.new(-75, -70, v_o_sm < 0 and o.o < 0 ? cgn : v_o_sm < 0 or o.o < 0 ? color.new(cgn, 40) : colnt, gds == 'Both' or gds == 'Bear')

float momFadeSource = (high + low + close) / 3.0
float momFadeBasis = ta.sma(momFadeSource, momFadeLen)
float momFadeDev = ta.stdev(momFadeSource, momFadeLen) * momFadeBbMult
float momFadeUpperBB = momFadeBasis + momFadeDev
float momFadeLowerBB = momFadeBasis - momFadeDev
float momFadeRange = momFadeUseTrueRange ? ta.tr(true) : high - low
float momFadeRangeMa = ta.sma(momFadeRange, momFadeLen) * momFadeKcMult
float momFadeUpperKC = momFadeBasis + momFadeRangeMa
float momFadeLowerKC = momFadeBasis - momFadeRangeMa
bool momFadeSqueezeOn = momFadeLowerBB > momFadeLowerKC and momFadeUpperBB < momFadeUpperKC
float momFadeLinReg = ta.linreg(momFadeSource, momFadeLen, 0)
float momFadeVal = momFadeLinReg - nz(momFadeLinReg[1], momFadeLinReg)
float momFadeHist = ta.linreg(momFadeVal, momFadeLen, 0)

// --- 진입 임계값 계산 ---
float dynSig = useDynamicThresh ? dynMult * ta.stdev(o.o, dynLen) : na
float baseThresh = math.abs(statThreshold)
float buyThreshActive  = -baseThresh
float sellThreshActive =  baseThresh
if useDynamicThresh
    float dynFallback = nz(dynSig, baseThresh)
    buyThreshActive  := -dynFallback
    sellThreshActive :=  dynFallback
else if not useSymThreshold
    buyThreshActive  := -math.abs(buyThreshold)
    sellThreshActive :=  math.abs(sellThreshold)


// 공통 ATR 캐시 (일관 호출 경고 방지용)
float atrLen_val = ta.atr(len)
// --- 기본 진입 신호 생성 ---
alerts a = alerts.new(ta.crossover(o.o, o.s) and o.o < buyThreshActive and v_o_sm > 0, ta.crossunder(o.o, o.s) and o.o > sellThreshActive and v_o_sm < 0, ta.crossover(o.o, 0), ta.crossunder(o.o, 0), ta.crossover(v_o_sm, 0), ta.crossunder(v_o_sm, 0), ta.crossunder(o.o, o.s) and smb, ta.crossover(o.o, o.s) and smb, ta.change(uG.c == colnt) and uG.c == color.new(cgp, 40), ta.change(dG.c == colnt) and dG.c == color.new(cgn, 40), ta.change(uG.c == colnt) and uG.c == cgp, ta.change(dG.c == colnt) and dG.c == cgn, ta.change(s.h) and s.h, ta.change(s.m) and s.m, ta.change(s.l) and s.l, p and o.o > trs, p and o.o < -trs)
bool baseLongSignal  = a.b
bool baseShortSignal = a.s


// --- 필터용 지표 계산 ---
// ADX/ATR (v5/v6 호환 & 일관 호출)
f_adx(len) =>
    [_, _, adx] = ta.dmi(len, len)
    adx

float adxValHtf = request.security(
     syminfo.tickerid, adxAtrTf,
     f_adx(adxLen), // 수정된 부분: 함수를 호출하여 ADX 값을 가져옵니다.
     lookahead = barmerge.lookahead_off)

adxValHtf := nz(adxValHtf)

// HTF ATR을 먼저 구한 뒤 동일 스코프에서 스무딩
float atrHtf = request.security(
     syminfo.tickerid, adxAtrTf,
     ta.atr(adxLen),
     lookahead = barmerge.lookahead_off)
float atrHtfSma = ta.sma(atrHtf, adxLen)
float atrDiffHtf = atrHtf - atrHtfSma
// 기타 필터 지표
float emaFast=ta.ema(close,emaFastLen)
float emaSlow=ta.ema(close,emaSlowLen)
float bbFilterBasis=ta.sma(close,bbLenFilter)
float bbFilterDev=ta.stdev(close,bbLenFilter)
float bbFilterUpper=bbFilterBasis+bbFilterDev*bbMultFilter
float bbFilterLower=bbFilterBasis-bbFilterDev*bbMultFilter
float rsiVal=ta.rsi(close,stochLen)
float rsiLow=ta.lowest(rsiVal,stochLen)
float rsiHigh=ta.highest(rsiVal,stochLen)
float stochRsiK=rsiHigh!=rsiLow?(rsiVal-rsiLow)/(rsiHigh-rsiLow)*100.0:50.0
float stochRsiVal=ta.sma(stochRsiK,3)
var float obvSeries=na
float dir=math.sign(ta.change(close))
obvSeries:=nz(obvSeries[1],0)+dir*nz(volume,0)
float obvSlope=ta.ema(ta.change(obvSeries),obvSmoothLen)
float htfMa=request.security(syminfo.tickerid,htfTrendTf,ta.ema(close,htfMaLen),lookahead=barmerge.lookahead_off)
bool htfTrendUp=close>htfMa
bool htfTrendDown=close<htfMa
float hmaValue=ta.hma(close,hmaLen)
float rangeHigh=request.security(syminfo.tickerid,rangeTf,ta.highest(high,rangeBars),lookahead=barmerge.lookahead_off)
float rangeLow=request.security(syminfo.tickerid,rangeTf,ta.lowest(low,rangeBars),lookahead=barmerge.lookahead_off)
float rangePerc=rangeLow!=0?(rangeHigh-rangeLow)/rangeLow*100.0:0.0
bool inRangeBox=rangePerc<=rangePercent
bool inSession = not na(time(timeframe.period, usSession))
bool eventBlock = false


// KCAS 컨텍스트 계산
float emaFastK=ta.ema(close,emaFastLenBase)
float emaSlowK=ta.ema(close,emaSlowLenBase)
bool microTrendLong=not useMicroTrend or emaFastK>emaSlowK
bool microTrendShort=not useMicroTrend or emaFastK<emaSlowK
float maTrend=ta.ema(close,trendLenBase)
float maConf=ta.ema(close,confLenBase)
bool trendBiasLongOK=not useTrendBias or close>maTrend
bool trendBiasShortOK=not useTrendBias or close<maTrend
bool confBiasLongOK=not useConfBias or close>maConf
bool confBiasShortOK=not useConfBias or close<maConf
float prevTrend=nz(maTrend[slopeLookback],maTrend)
float slopePct=maTrend!=0?(maTrend-prevTrend)/maTrend*100.0:0.0
bool slopeOK_L=not useSlopeFilter or slopePct>=slopeMinPct
bool slopeOK_S=not useSlopeFilter or slopePct<=-slopeMinPct
float distanceAtr=ta.atr(distanceAtrLen)
float vwDistance=distanceAtr>0?math.abs(close-ta.vwap)/distanceAtr:0.0
float trendDistance=distanceAtr>0?math.abs(close-maTrend)/distanceAtr:0.0
bool distanceOK_L=not useDistanceGuard or (vwDistance<=distanceMaxAtr and trendDistance<=distanceMaxAtr)
bool distanceOK_S=distanceOK_L
float ctxHtfAdx=request.security(syminfo.tickerid,ctxHtfTf, f_adx(ctxHtfAdxLen), lookahead=barmerge.lookahead_off) // [수정] 두 번째 오류 수정
float ctxHtfEma=request.security(syminfo.tickerid,ctxHtfTf,ta.ema(close,ctxHtfEmaLen),lookahead=barmerge.lookahead_off)
bool htfLongOK=not useRegimeFilter or (close>ctxHtfEma[1] and nz(ctxHtfAdx[1])>ctxHtfAdxTh)
bool htfShortOK=not useRegimeFilter or (close<ctxHtfEma[1] and nz(ctxHtfAdx[1])>ctxHtfAdxTh)
float eqSlope=ta.linreg(strategy.equity,eqSlopeLen,0)-ta.linreg(strategy.equity,eqSlopeLen,1)
bool equitySlopeOK_L=not useEquitySlopeFilter or eqSlope>=0
bool equitySlopeOK_S=not useEquitySlopeFilter or eqSlope<=0

// --- 최종 진입 조건 결합 ---
bool longOk=true
longOk := longOk and (not useAdx or adxValHtf>adxThresh) and (not useEma or (emaMode=="Crossover"?emaFast>emaSlow:close>emaSlow)) and (not useBb or close<bbFilterLower or close<=bbFilterBasis) and (not useStochRsi or stochRsiVal<=stochOS) and (not useObv or obvSlope>0) and (not useAtrDiff or atrDiffHtf>0) and (not useHtfTrend or htfTrendUp) and (not useHmaFilter or close>hmaValue) and (not useRangeFilter or not inRangeBox) and (not useSessionFilter or inSession) and (not useEventFilter or not eventBlock)
longOk := longOk and microTrendLong and trendBiasLongOK and confBiasLongOK and slopeOK_L and distanceOK_L and htfLongOK and equitySlopeOK_L
bool shortOk=true
shortOk := shortOk and (not useAdx or adxValHtf>adxThresh) and (not useEma or (emaMode=="Crossover"?emaFast<emaSlow:close<emaSlow)) and (not useBb or close>bbFilterUpper or close>=bbFilterBasis) and (not useStochRsi or stochRsiVal>=stochOB) and (not useObv or obvSlope<0) and (not useAtrDiff or atrDiffHtf>0) and (not useHtfTrend or htfTrendDown) and (not useHmaFilter or close<hmaValue) and (not useRangeFilter or not inRangeBox) and (not useSessionFilter or inSession) and (not useEventFilter or not eventBlock)
shortOk := shortOk and microTrendShort and trendBiasShortOK and confBiasShortOK and slopeOK_S and distanceOK_S and htfShortOK and equitySlopeOK_S

bool longStructPass = structureGateMode=="All"?((not useBOS or bosLong_vn) and (not useCHOCH or chochLong_state)):((useBOS and bosLong_vn) or (useCHOCH and chochLong_state) or (not useBOS and not useCHOCH))
bool shortStructPass = structureGateMode=="All"?((not useBOS or bosShort_vn) and (not useCHOCH or chochShort_state)):((useBOS and bosShort_vn) or (useCHOCH and chochShort_state) or (not useBOS and not useCHOCH))
bool longGateOk = (not useSqzGate or relOk_vn) and (not useVSpike or true) and longStructPass and canTrade
bool shortGateOk = (not useSqzGate or relOk_vn) and (not useVSpike or true) and shortStructPass and canTrade
longOk := longOk and longGateOk
shortOk := shortOk and shortGateOk

bool enterLong = (allowLongEntry and baseLongSignal) and (useScore_vn?false:longOk)
bool enterShort = (allowShortEntry and baseShortSignal) and (useScore_vn?false:shortOk)

// --- 재진입 및 반대매매 로직 ---
bool justClosed=strategy.position_size[1]!=0 and strategy.position_size==0
var int reentryCountdown=0
if justClosed
    reentryCountdown:=reentryBars
else if strategy.position_size==0 and reentryCountdown>0
    reentryCountdown:=reentryCountdown-1
var int reversalCountdown=0
var int lastPosDir=0
bool posClosedThisBar=strategy.position_size[1]!=0 and strategy.position_size==0
if posClosedThisBar
    if useReversal
        float barSec=(time-time[1])/1000.0
        int delayBars=barSec>0?math.round(reversalDelaySec/barSec):0
        reversalCountdown:=delayBars
        lastPosDir:=strategy.position_size[1]>0?1:-1
    else
        reversalCountdown:=0
        lastPosDir:=0
else if reversalCountdown>0
    reversalCountdown:=reversalCountdown-1
var bool reversalLongSignal=false
var bool reversalShortSignal=false
reversalLongSignal:=false
reversalShortSignal:=false
if useReversal and reversalCountdown==0 and strategy.position_size==0 and lastPosDir!=0
    if lastPosDir==1
        reversalShortSignal:=true
    else if lastPosDir==-1
        reversalLongSignal:=true
    lastPosDir:=0
if reversalLongSignal and canTrade
    enterLong:=true
if reversalShortSignal and canTrade
    enterShort:=true
enterLong:=enterLong and (reentryCountdown==0)
enterShort:=enterShort and (reentryCountdown==0)


// --- 청산 조건 계산 ---
int momFadeSignNow = int(math.sign(momFadeHist))
int momFadeSignPrev = int(math.sign(nz(momFadeHist[1], momFadeHist)))
var int momFadeEntryDir = 0
var int momFadeEntrySign = 0
var bool momFadeFlipArmed = false
var int momFadeZeroCrossBar = na
var int momFadeLastOppBar = na
var int momFadeReleaseBar = na

bool newLongPosition = strategy.position_size > 0 and strategy.position_size[1] <= 0
bool newShortPosition = strategy.position_size < 0 and strategy.position_size[1] >= 0
bool positionClosed = strategy.position_size == 0 and strategy.position_size[1] != 0
if newLongPosition
    momFadeEntryDir := 1
    int initSign = momFadeSignNow != 0 ? momFadeSignNow : momFadeSignPrev != 0 ? momFadeSignPrev : -1
    momFadeEntrySign := initSign
    momFadeFlipArmed := false
    momFadeZeroCrossBar := na
    momFadeLastOppBar := na
    momFadeReleaseBar := na
else if newShortPosition
    momFadeEntryDir := -1
    int initSign = momFadeSignNow != 0 ? momFadeSignNow : momFadeSignPrev != 0 ? momFadeSignPrev : 1
    momFadeEntrySign := initSign
    momFadeFlipArmed := false
    momFadeZeroCrossBar := na
    momFadeLastOppBar := na
    momFadeReleaseBar := na
else if positionClosed
    momFadeEntryDir := 0
    momFadeEntrySign := 0
    momFadeFlipArmed := false
    momFadeZeroCrossBar := na
    momFadeLastOppBar := na
    momFadeReleaseBar := na
else if strategy.position_size != 0
    if momFadeEntrySign == 0 and momFadeSignNow != 0
        momFadeEntrySign := momFadeSignNow
    bool isOppNow = momFadeEntrySign != 0 and momFadeSignNow == -momFadeEntrySign and momFadeSignNow != 0
    bool crossOpp = isOppNow and momFadeSignPrev != momFadeSignNow
    if crossOpp
        momFadeFlipArmed := true
        momFadeZeroCrossBar := bar_index
        momFadeReleaseBar := not momFadeSqueezeOn ? bar_index : na
    if isOppNow
        momFadeLastOppBar := na(momFadeLastOppBar) ? bar_index : momFadeLastOppBar
    else if momFadeSignNow == 0
        momFadeLastOppBar := na
    if momFadeFlipArmed and momFadeReleaseBar == na and not momFadeSqueezeOn
        momFadeReleaseBar := bar_index
    if momFadeSignNow == momFadeEntrySign and momFadeSignPrev == -momFadeEntrySign and momFadeSignNow != 0
        momFadeFlipArmed := false
        momFadeZeroCrossBar := na
        momFadeLastOppBar := na
        momFadeReleaseBar := na

var int posBars=0
posBars:=strategy.position_size!=0?posBars+1:0
bool exitLongOpposite=exitOpposite and baseShortSignal and posBars>=minHoldBars
bool exitShortOpposite=exitOpposite and baseLongSignal and posBars>=minHoldBars
bool exitLongStoch=useStochRsiExit and stochRsiVal>=stochOB and posBars>=minHoldBars
bool exitShortStoch=useStochRsiExit and stochRsiVal<=stochOS and posBars>=minHoldBars
int fadeBars = math.max(momFadeBars, 1)
float momFadeAbs = math.abs(momFadeHist)
float momFadeAbsPrev = math.abs(nz(momFadeHist[1], momFadeHist))
bool fadeMagnitudeDown = fadeBars <= 1 ? momFadeAbs < momFadeAbsPrev : ta.falling(momFadeAbs, fadeBars)
bool fadeOppActive = momFadeFlipArmed and momFadeEntrySign != 0 and momFadeSignNow == -momFadeEntrySign and momFadeSignNow != 0
bool fadeDelayOk = momFadeZeroCrossBar != na and (bar_index - momFadeZeroCrossBar) >= momFadeZeroDelay
bool fadeMinAbsOk = momFadeMinAbs <= 0 or momFadeAbs >= momFadeMinAbs
bool fadeWindowOk = momFadeLastOppBar != na
bool fadeReleaseOk = momFadeReleaseBar != na
bool exitLongFade=useMomFade and momFadeEntryDir == 1 and fadeOppActive and fadeMagnitudeDown and fadeDelayOk and fadeMinAbsOk and fadeWindowOk and fadeReleaseOk and posBars>=minHoldBars
bool exitShortFade=useMomFade and momFadeEntryDir == -1 and fadeOppActive and fadeMagnitudeDown and fadeDelayOk and fadeMinAbsOk and fadeWindowOk and fadeReleaseOk and posBars>=minHoldBars
float kasaRsi = ta.rsi(close, kasa_rsiLen)
bool kasaExitLong = useKASA and ta.crossunder(kasaRsi, kasa_rsiOB)
bool kasaExitShort = useKASA and ta.crossover(kasaRsi, kasa_rsiOS)

// --- [복원] 쇼크 필터 계산 ---
float atrFast = ta.atr(atrFastLen)
float atrSlow = ta.sma(atrFast, atrSlowLen)
bool isShock = useShock and (atrFast > atrSlow * shockMult)

// --- 손익절 라인 계산 ---
float atrTrail=ta.atr(atrTrailLen)
float dynAtrRatio=atrTrail/close
float dynBBDev=2.0*ta.stdev(close,20)
float dynBBWidth=(dynBBDev*2.0)/close
float dynMa50=ta.sma(close,50)
float dynMaDist=math.abs(close-dynMa50)/close
float dynMetric=(dynAtrRatio+dynBBWidth+dynMaDist)/3.0
float dynRaw=1.0+dynMetric
float dynFactor=useDynVol?math.max(0.5,math.min(3.0,dynRaw)):1.0
float trailDist=atrTrail*atrTrailMult*dynFactor
float highestHigh=ta.highest(high,atrTrailLen)
float lowestLow=ta.lowest(low,atrTrailLen)
float trailStopLong=highestHigh-trailDist
float trailStopShort=lowestLow+trailDist
var float pivotHighHtf=na
var float pivotLowHtf=na
pivotHighHtf:=request.security(syminfo.tickerid,pivotTf,ta.pivothigh(high,pivotLen,pivotLen),lookahead=barmerge.lookahead_off)
pivotLowHtf:=request.security(syminfo.tickerid,pivotTf,ta.pivotlow(low,pivotLen,pivotLen),lookahead=barmerge.lookahead_off)
float ph=na
float pl=na
if usePivotHtf
    ph:=pivotHighHtf
    pl:=pivotLowHtf
else
    [ph,pl]=getPivots(pivotLen)
if not na(ph)
    pivotHighStop:=ph
if not na(pl)
    pivotLowStop:=pl
float pivotStopLong=(usePivotStop and not na(pivotLowStop))?pivotLowStop:na
float pivotStopShort=(usePivotStop and not na(pivotHighStop))?pivotHighStop:na
float baseStopLong=useStopLoss?(not na(pivotStopLong)?pivotStopLong:ta.lowest(low,stopLookback)):na
float baseStopShort=useStopLoss?(not na(pivotStopShort)?pivotStopShort:ta.highest(high,stopLookback)):na
float finalStopLong=na
float finalStopShort=na
float tempLong=na
if useAtrTrail and not na(trailStopLong)
    tempLong:=na(tempLong)?trailStopLong:math.max(tempLong,trailStopLong)
if useStopLoss and not na(baseStopLong)
    tempLong:=na(tempLong)?baseStopLong:math.max(tempLong,baseStopLong)
if usePivotStop and not na(pivotStopLong)
    tempLong:=na(tempLong)?pivotStopLong:math.max(tempLong,pivotStopLong)
finalStopLong:=tempLong
float tempShort=na
if useAtrTrail and not na(trailStopShort)
    tempShort:=na(tempShort)?trailStopShort:math.min(tempShort,trailStopShort)
if useStopLoss and not na(baseStopShort)
    tempShort:=na(tempShort)?baseStopShort:math.min(tempShort,baseStopShort)
if usePivotStop and not na(pivotStopShort)
    tempShort:=na(tempShort)?pivotStopShort:math.min(tempShort,pivotStopShort)
finalStopShort:=tempShort
var float highestSinceEntry=na
var float lowestSinceEntry=na
if strategy.position_size!=0
    bool isNewPos=strategy.position_size[1]==0
    if strategy.position_size>0
        highestSinceEntry:=isNewPos?high:math.max(nz(highestSinceEntry,high),high)
        lowestSinceEntry:=na
    else if strategy.position_size<0
        lowestSinceEntry:=isNewPos?low:math.min(nz(lowestSinceEntry,low),low)
        highestSinceEntry:=na
else
    highestSinceEntry:=na
    lowestSinceEntry:=na


// =================================================================================
// === 실행 (Execution) & 시각화 (Plotting) =======================================
// =================================================================================
// --- 전략 실행 ---
if time >= startDate
    if strategy.position_size == 0
        if enterLong
            float stopHintLong = na
            if useStopLoss
                float swingLow = ta.lowest(low, stopLookback)
                if not na(swingLow)
                    float dist = close - swingLow
                    stopHintLong := na(stopHintLong) ? dist : math.max(stopHintLong, dist)
            if useAtrTrail
                float atrDist = ta.atr(atrTrailLen) * atrTrailMult
                stopHintLong := na(stopHintLong) ? atrDist : math.max(stopHintLong, atrDist)
            if usePivotStop
                float pivotRef = usePivotHtf ? nz(pivotLowHtf, low) : ta.lowest(low, pivotLen)
                if not na(pivotRef)
                    float distPivot = close - pivotRef
                    stopHintLong := na(stopHintLong) ? distPivot : math.max(stopHintLong, distPivot)
            if na(stopHintLong)
                stopHintLong := atrLen_val
            float stopForSizeL = na(stopHintLong) ? tickSize : math.max(stopHintLong, tickSize)
            float atrGuardRefL = atrLen_val
            bool stopGuardOkL = not useStopDistanceGuard or na(atrGuardRefL) or atrGuardRefL == 0.0 or stopForSizeL <= atrGuardRefL * maxStopAtrMult
            float qty = calcOrderSize(close, stopForSizeL, 1.0)
            if qty > 0 and stopGuardOkL
                strategy.entry("Long",strategy.long,qty=qty,alert_message=alertLongEntry)
            if useAtrProfit and not usePartialProfit
                strategy.exit("LongProfit",from_entry="Long",limit=close+atrTrail*atrProfitMult*dynFactor)
        else if enterShort
            float stopHintShort = na
            if useStopLoss
                float swingHigh = ta.highest(high, stopLookback)
                if not na(swingHigh)
                    float dist = swingHigh - close
                    stopHintShort := na(stopHintShort) ? dist : math.max(stopHintShort, dist)
            if useAtrTrail
                float atrDistS = ta.atr(atrTrailLen) * atrTrailMult
                stopHintShort := na(stopHintShort) ? atrDistS : math.max(stopHintShort, atrDistS)
            if usePivotStop
                float pivotRefS = usePivotHtf ? nz(pivotHighHtf, high) : ta.highest(high, pivotLen)
                if not na(pivotRefS)
                    float distPivotS = pivotRefS - close
                    stopHintShort := na(stopHintShort) ? distPivotS : math.max(stopHintShort, distPivotS)
            if na(stopHintShort)
                stopHintShort := atrLen_val
            float stopForSizeS = na(stopHintShort) ? tickSize : math.max(stopHintShort, tickSize)
            float atrGuardRefS = atrLen_val
            bool stopGuardOkS = not useStopDistanceGuard or na(atrGuardRefS) or atrGuardRefS == 0.0 or stopForSizeS <= atrGuardRefS * maxStopAtrMult
            float qty = calcOrderSize(close, stopForSizeS, 1.0)
            if qty > 0 and stopGuardOkS
                strategy.entry("Short",strategy.short,qty=qty,alert_message=alertShortEntry)
            if useAtrProfit and not usePartialProfit
                strategy.exit("ShortProfit",from_entry="Short",limit=close-atrTrail*atrProfitMult*dynFactor)
    else if strategy.position_size > 0
        bool exitLongTime=useTimeStop and (maxHoldBars>0) and (posBars>=maxHoldBars)
        float stopLongToUse=finalStopLong
        if isShock and shockAction == "손절 타이트닝"
            float shockStop = low[1]
            stopLongToUse := na(stopLongToUse) ? shockStop : math.max(stopLongToUse, shockStop)
        if isShock and shockAction == "즉시 청산"
            strategy.close("Long", comment="변동성 쇼크 청산")
        if exitLongOpposite or exitLongStoch or exitLongFade or exitLongTime or kasaExitLong
            strategy.close("Long",alert_message=alertExitLong)
        if useBreakevenStop
            if not na(highestSinceEntry) and (highestSinceEntry-strategy.position_avg_price)>=(atrTrail*breakevenMult*dynFactor)
                stopLongToUse:=na(stopLongToUse)?strategy.position_avg_price:math.max(stopLongToUse,strategy.position_avg_price)
        if useBETiers and not na(highestSinceEntry)
            float atrSeed = atrLen_val
            if atrSeed > 0 and (highestSinceEntry - strategy.position_avg_price) >= atrSeed
                stopLongToUse := na(stopLongToUse) ? strategy.position_avg_price : math.max(stopLongToUse, strategy.position_avg_price)
        if not na(stopLongToUse)
            strategy.exit("LongStop",from_entry="Long",stop=stopLongToUse,alert_message=alertExitLong)
    else if strategy.position_size < 0
        bool exitShortTime=useTimeStop and (maxHoldBars>0) and (posBars>=maxHoldBars)
        float stopShortToUse=finalStopShort
        if isShock and shockAction == "손절 타이트닝"
            float shockStop = high[1]
            stopShortToUse := na(stopShortToUse) ? shockStop : math.min(stopShortToUse, shockStop)
        if isShock and shockAction == "즉시 청산"
            strategy.close("Short", comment="변동성 쇼크 청산")
        if exitShortOpposite or exitShortStoch or exitShortFade or exitShortTime or kasaExitShort
            strategy.close("Short",alert_message=alertExitShort)
        if useBreakevenStop
            if not na(lowestSinceEntry) and (strategy.position_avg_price-lowestSinceEntry)>=(atrTrail*breakevenMult*dynFactor)
                stopShortToUse:=na(stopShortToUse)?strategy.position_avg_price:math.min(stopShortToUse,strategy.position_avg_price)
        if useBETiers and not na(lowestSinceEntry)
            float atrSeedS = atrLen_val
            if atrSeedS > 0 and (strategy.position_avg_price - lowestSinceEntry) >= atrSeedS
                stopShortToUse := na(stopShortToUse) ? strategy.position_avg_price : math.min(stopShortToUse, strategy.position_avg_price)
        if not na(stopShortToUse)
            strategy.exit("ShortStop",from_entry="Short",stop=stopShortToUse,alert_message=alertExitShort)

// --- 시각화 ---
color colsq = s.h ? colsh : s.m ? colsm : colsl
color colvf = v_o_sm > 0 ? color.new(cps, 70) : color.new(cng, 70)
color colof = v_s_sm > 0 ? color.new(cpo, 70) : color.new(cno, 70)
color colsf = o.o > o.s ? color.new(cpf, 50) : color.new(cdf, 50)
color colzf = o.o > o.s ? cup : cdn
hline(0,title="Mid-Line",color=color.new(color.white,70),linestyle=hline.style_dashed,linewidth=1)
plot(dfb?v.o:na,title="Directional Flux",color=colvf,linewidth=1,style=plot.style_area)
plot(dfb?v.s:na,title="OverFlux",color=colof,linewidth=1,style=plot.style_areabr)
plot(smb?o.o:na,title="Momentum",color=colzf,linewidth=1,style=plot.style_line)
plot(smb?o.s:na,title="Momentum Signal",display=display.none)
plot(s.l?1:na,title="Squeeze Level",color=colsq,linewidth=1,style=plot.style_columns,display=display.pane)
plot(a.x?o.s:na,title="Bearish Swing",color=cdf,linewidth=2,style=plot.style_circles,display=display.pane)
plot(a.y?o.o:na,title="Bullish Swing",color=cpf,linewidth=2,style=plot.style_circles,display=display.pane)
fill(plot(dG.p?dG.l:na,display=display.none),plot(dG.p?dG.u:na,display=display.none),dG.c)
fill(plot(uG.p?uG.l:na,display=display.none),plot(uG.p?uG.u:na,display=display.none),uG.c)
plotshape(dbs and a.e?o.s+3:na,title="Bearish Divergence",style=shape.labeldown,location=location.absolute,color=colnt,text='𝐃▾',textcolor=colsm)
plotshape(dbs and a.f?o.s-3:na,title="Bullish Divergence",style=shape.labelup,location=location.absolute,color=colnt,text='𝐃▴',textcolor=colsm)
plotshape(uG.p and a.s?uG.u+10:na,title="Confluence Sell",style=shape.triangledown,location=location.absolute,color=colno,size=size.tiny)
plotshape(dG.p and a.b?dG.l-15:na,title="Confluence Buy",style=shape.triangleup,location=location.absolute,color=colpo,size=size.tiny)
plotchar(isShock, title="변동성 쇼크", char="⚡", location=location.top, color=color.new(color.yellow, 0), size=size.tiny)

// --- KCAS HUD & 디버거 ---
if showHudPanel and barstate.islast
    var table hud = table.new(hudPosition == "Top Left" ? position.top_left : hudPosition == "Bottom Left" ? position.bottom_left : hudPosition == "Bottom Right" ? position.bottom_right : position.top_right, 2, 12, bgcolor=color.new(color.black, 45), border_width=1, border_color=color.gray)
    table.cell(hud, 0, 0, "KCAS HUD", text_color=color.white, bgcolor=color.new(color.purple, 35))
    table.cell(hud, 1, 0, "", bgcolor=color.new(color.purple, 35))
    table.merge_cells(hud, 0, 0, 1, 0)
    table.cell(hud, 0, 1, "거래 가능 자본", text_color=color.white)
    table.cell(hud, 1, 1, str.tostring(tradableCapital, format.mintick), text_color=color.aqua)
    table.cell(hud, 0, 2, "적립된 수익", text_color=color.white)
    table.cell(hud, 1, 2, str.tostring(withdrawable, format.mintick), text_color=color.yellow)
    string dailyTargetTxt = useDailyProfitLock ? str.tostring(dailyProfitTarget, format.mintick) : "OFF"
    color dailyBg = dailyProfitReached ? color.new(color.lime, 60) : dailyPnl >= 0 ? color.new(color.aqua, 70) : color.new(color.red, 70)
    table.cell(hud, 0, 3, "일일 PnL", text_color=color.white)
    table.cell(hud, 1, 3, str.tostring(dailyPnl, format.mintick) + " / " + dailyTargetTxt, text_color=color.white, bgcolor=dailyBg)
    string weeklyTargetTxt = useWeeklyProfitLock ? str.tostring(weeklyProfitTarget, format.mintick) : "OFF"
    color weeklyBg = weeklyProfitReached ? color.new(color.lime, 60) : weeklyPnl >= 0 ? color.new(color.aqua, 70) : color.new(color.orange, 70)
    table.cell(hud, 0, 4, "주간 PnL", text_color=color.white)
    table.cell(hud, 1, 4, str.tostring(weeklyPnl, format.mintick) + " / " + weeklyTargetTxt, text_color=color.white, bgcolor=weeklyBg)
    table.cell(hud, 0, 5, "ATR%", text_color=color.white)
    table.cell(hud, 1, 5, str.tostring(atrPct, "##.##") + "%", text_color=isVolatilityOK ? color.aqua : color.red)
    table.cell(hud, 0, 6, "가드 상태", text_color=color.white)
    table.cell(hud, 1, 6, haltReasons ? "중지" : "가동", text_color=color.white, bgcolor=color.new(haltReasons ? color.red : color.green, 55))
    table.cell(hud, 0, 7, "PAR", text_color=color.white)
    table.cell(hud, 1, 7, parStateLabel + " / " + parWinLabel + " / " + str.tostring(finalRiskPct, "#.##") + "%", text_color=color.white)
    table.cell(hud, 0, 8, "연패", text_color=color.white)
    table.cell(hud, 1, 8, str.tostring(lossStreak), text_color=lossStreakBreached ? color.red : color.white)
    string dailyLossTxt = maxDailyLosses > 0 ? str.tostring(dailyLosses) + "/" + str.tostring(maxDailyLosses) : str.tostring(dailyLosses) + "/∞"
    table.cell(hud, 0, 9, "일일 손실", text_color=color.white)
    table.cell(hud, 1, 9, dailyLossTxt, text_color=color.white, bgcolor=lossCountBreached ? color.new(color.red, 60) : color.new(color.black, 0))
    string weeklyDdTxt = maxWeeklyDD > 0 ? str.tostring(weeklyDD, "##.##") + "% / " + str.tostring(maxWeeklyDD, "##.##") + "%" : str.tostring(weeklyDD, "##.##") + "% / ∞"
    table.cell(hud, 0, 10, "주간 DD", text_color=color.white)
    table.cell(hud, 1, 10, weeklyDdTxt, text_color=color.white, bgcolor=weeklyDDBreached ? color.new(color.red, 60) : color.new(color.black, 0))
    string guardCountTxt = maxGuardFires > 0 ? str.tostring(guardFiredTotal) + "/" + str.tostring(maxGuardFires) : str.tostring(guardFiredTotal) + "/∞"
    table.cell(hud, 0, 11, "가드 카운트", text_color=color.white)
    table.cell(hud, 1, 11, guardCountTxt, text_color=color.white, bgcolor=guardFireLimit ? color.new(color.red, 60) : color.new(color.black, 0))

if showDebugPanel and barstate.islast
    var table dbg = table.new(position.bottom_right, 3, 8, bgcolor=color.new(color.black, 45), border_width=1, border_color=color.gray)
    table.cell(dbg, 0, 0, "디버그", text_color=color.white, bgcolor=color.new(color.blue, 35))
    table.cell(dbg, 1, 0, "", bgcolor=color.new(color.blue, 35))
    table.cell(dbg, 2, 0, "", bgcolor=color.new(color.blue, 35))
    table.merge_cells(dbg, 0, 0, 2, 0)
    table.cell(dbg, 0, 1, "시간", text_color=color.white)
    table.cell(dbg, 1, 1, str.tostring(sessionAllowed and kstAllowed and dayAllowed), text_color=color.white, bgcolor=color.new((sessionAllowed and kstAllowed and dayAllowed) ? color.green : color.red, 70))
    table.cell(dbg, 0, 2, "레짐 L/S", text_color=color.white)
    table.cell(dbg, 1, 2, str.tostring(htfLongOK), text_color=htfLongOK ? color.aqua : color.red)
    table.cell(dbg, 2, 2, str.tostring(htfShortOK), text_color=htfShortOK ? color.orange : color.red)
    table.cell(dbg, 0, 3, "모멘텀", text_color=color.white)
    table.cell(dbg, 1, 3, str.tostring(baseLongSignal), text_color=baseLongSignal ? color.aqua : color.white)
    table.cell(dbg, 2, 3, str.tostring(baseShortSignal), text_color=baseShortSignal ? color.orange : color.white)
    table.cell(dbg, 0, 4, "컨텍스트", text_color=color.white)
    table.cell(dbg, 1, 4, str.tostring(microTrendLong and trendBiasLongOK and distanceOK_L), text_color=color.white)
    table.cell(dbg, 2, 4, str.tostring(microTrendShort and trendBiasShortOK and distanceOK_S), text_color=color.white)
    table.cell(dbg, 0, 5, "가드", text_color=color.white)
    table.cell(dbg, 1, 5, str.tostring(guardFrozen), text_color=guardFrozen ? color.red : color.white)
    table.cell(dbg, 2, 5, str.tostring(guardFiredTotal), text_color=guardFiredTotal > 0 ? color.orange : color.white)
    table.cell(dbg, 0, 6, "WinRate", text_color=color.white)
    table.cell(dbg, 1, 6, parWinLabel, text_color=color.white)
    table.cell(dbg, 0, 7, "canTrade", text_color=color.white)
    table.cell(dbg, 1, 7, str.tostring(canTrade), text_color=color.white, bgcolor=color.new(canTrade ? color.green : color.red, 70))

// --- 얼럿 조건 ---
alertcondition(a.s, "Confluence Sell", "Sell Signal")
alertcondition(a.b, "Confluence Buy", "Buy Signal")
alertcondition(a.u, "Momentum Midline Crossover", "Momentum Bullish")
alertcondition(a.d, "Momentum Midline Crossunder", "Momentum Bearish")
alertcondition(a.p, "Flux Midline Crossover", "Flux Bullish")
alertcondition(a.n, "Flux Midline Crossunder", "Flux Bearish")
alertcondition(a.y, "Momentum Swing Crossover", "Bullish Swing")
alertcondition(a.x, "Momentum Swing Crossunder", "Bearish Swing")
alertcondition(a.q, "Strong Bullish Confluence", "Strong Bullish Confluence")
alertcondition(a.w, "Strong Bearish Confluence", "Strong Bearish Confluence")
alertcondition(a.a, "Weak Bullish Confluence", "Weak Bullish Confluence")
alertcondition(a.c, "Weak Bearish Confluence", "Weak Bearish Confluence")
alertcondition(a.h, "High Squeeze", "High Squeeze")
alertcondition(a.m, "Normal Squeeze", "Normal Squeeze")
alertcondition(a.l, "Low Squeeze", "Low Squeeze")
alertcondition(a.e, "Bearish Divergence", "Bearish Divergence")
alertcondition(a.f, "Bullish Divergence", "Bullish Divergence")





2. 2번스크립트 SMC기반


//@version=5
// [MERGE] inserted to avoid undefined vars
// PPP + BOS + OB + FVG + CHoCH Strategy (1–3m scalping tuned)
// — KASIA patch v3d: 
//    • Heikin-Ashi fix (ticker.heikinashi)
//    • ADX: 자유 TF + 전역 precompute
//    • HTF Trend / Range / MA 모두 자유 TF로 통일 (request.security는 전역·무조건 호출)
//    • Daily loss stop: 전역 호출
//    • webhook 제거 유지
//
// Designed for: Crypto 1–3m charts. Core SMC preserved.
//
// ─────────────────────────────────────────────────────────────────────────────

strategy(
     title               = "KASIA_SMC_Scalp_v3d (no-webhook)",
     overlay             = true,
     max_bars_back       = 5000,
     initial_capital     = 500,
     pyramiding          = 0,
     default_qty_type    = strategy.percent_of_equity,
     default_qty_value   = 50,
     commission_type     = strategy.commission.percent,
     commission_value    = 0.05,
     calc_on_every_tick  = true
     )

// ─── General ───
g_general     = "General Settings"
orderPercent = input.float(15.0, "주문 비율 (%)", minval=1, maxval=100, group=g_general)
leverage      = input.int  (20,   "레버리지(정보용)",              minval=1, maxval=100, group=g_general)

// ─── Timeframe Settings (모두 자유 TF) ───
g_tf    = "Timeframe Settings"
htf1    = input.timeframe("5",  "상위TF 1 (추세)",    group=g_tf)   // 자유 입력
htf2    = input.timeframe("15", "상위TF 2 (추세)",   group=g_tf)   // 자유 입력

// State vars
var float entryPrice      = na
var float tp1             = na
var float tp2             = na
var bool  isLongTp1Hit    = false
var float tp1s            = na
var bool  isShortTp1Hit   = false
var float tp2s            = na
var int   entryBarIndex   = na

// ─── Trailing/BE state ───
// trailing states: use slLongFrozen/slShortFrozen only; remove unused vars
var float slLongFrozen = na
var float slShortFrozen = na
// last structural stop for structure-based exit
var float lastStructuralStop = na

g_weights     = "Condition Weights"

// [가중치 모드 프리셋] — 보수/중립/공격/초공격/커스텀
__kasia_mode = input.string("커스텀", "가중치 모드", options=["보수","중립","공격","초공격","커스텀"], group="[가중치/프리셋]")
weightFVG_custom = input.int(2, "FVG 가중치",      minval=0, maxval=5, group=g_weights)

weightFVG = __kasia_mode == "커스텀" ? weightFVG_custom : (__kasia_mode == "보수" ? 2 : __kasia_mode == "공격" ? 2 : __kasia_mode == "초공격" ? 2 : 2)
weightOB_custom = input.int(1, "OB 가중치",       minval=0, maxval=5, group=g_weights)

weightOB = __kasia_mode == "커스텀" ? weightOB_custom : (__kasia_mode == "보수" ? 2 : __kasia_mode == "공격" ? 1 : __kasia_mode == "초공격" ? 1 : 2)
weightBOS_custom = input.int(2, "BOS 가중치",      minval=0, maxval=5, group=g_weights)

weightBOS = __kasia_mode == "커스텀" ? weightBOS_custom : (__kasia_mode == "보수" ? 3 : __kasia_mode == "공격" ? 2 : __kasia_mode == "초공격" ? 2 : 3)
weightCHoCH_custom = input.int(1, "CHoCH 가중치",    minval=0, maxval=5, group=g_weights)

weightCHoCH = __kasia_mode == "커스텀" ? weightCHoCH_custom : (__kasia_mode == "보수" ? 1 : __kasia_mode == "공격" ? 1 : __kasia_mode == "초공격" ? 1 : 1)
weightPivot_custom = input.int(1, "피벗 가중치",    minval=0, maxval=5, group=g_weights)

weightPivot = __kasia_mode == "커스텀" ? weightPivot_custom : (__kasia_mode == "보수" ? 1 : __kasia_mode == "공격" ? 1 : __kasia_mode == "초공격" ? 1 : 1)
weightTrend_custom = input.int(5, "추세 가중치",    minval=0, maxval=5, group=g_weights)

weightTrend = __kasia_mode == "커스텀" ? weightTrend_custom : (__kasia_mode == "보수" ? 5 : __kasia_mode == "공격" ? 3 : __kasia_mode == "초공격" ? 3 : 4)
weightMom_custom = input.int(4, "모멘텀 가중치", minval=0, maxval=5, group=g_weights)

weightMom = __kasia_mode == "커스텀" ? weightMom_custom : (__kasia_mode == "보수" ? 5 : __kasia_mode == "공격" ? 3 : __kasia_mode == "초공격" ? 3 : 4)
weightBB_custom = input.int(1, "브레이커블록 가중치",       minval=0, maxval=5, group=g_weights)


weightBB = __kasia_mode == "커스텀" ? weightBB_custom : (__kasia_mode == "보수" ? 1 : __kasia_mode == "공격" ? 1 : __kasia_mode == "초공격" ? 1 : 1)
g_threshold   = "Scoring Threshold"
minScoreLong_custom = input.int(9, "롱 최소 점수",  minval=1, maxval=20, group=g_threshold)  
minScoreLong = __kasia_mode == "커스텀" ? minScoreLong_custom : (__kasia_mode == "보수" ? 9 : __kasia_mode == "공격" ? 7 : __kasia_mode == "초공격" ? 6 : 8)
// tuned for 1m
minScoreShort_custom = input.int(9, "숏 최소 점수", minval=1, maxval=20, group=g_threshold)

minScoreShort = __kasia_mode == "커스텀" ? minScoreShort_custom : (__kasia_mode == "보수" ? 9 : __kasia_mode == "공격" ? 7 : __kasia_mode == "초공격" ? 6 : 8)
// ─────────────────────────────────────────────────────────────────────────────
// Enhancement Weights
weightLiqVoid_custom = input.int(1, "유동성 공백 가중치",      minval=0, maxval=5, group=g_weights)

weightLiqVoid = __kasia_mode == "커스텀" ? weightLiqVoid_custom : (__kasia_mode == "보수" ? 1 : __kasia_mode == "공격" ? 1 : __kasia_mode == "초공격" ? 1 : 1)
weightDiv_custom = input.int(1, "볼륨 다이버전스 가중치",  minval=0, maxval=5, group=g_weights)

weightDiv = __kasia_mode == "커스텀" ? weightDiv_custom : (__kasia_mode == "보수" ? 1 : __kasia_mode == "공격" ? 1 : __kasia_mode == "초공격" ? 1 : 1)
weightPattern_custom = input.int(1, "캔들 패턴 가중치",        minval=0, maxval=5, group=g_weights)

weightPattern = __kasia_mode == "커스텀" ? weightPattern_custom : (__kasia_mode == "보수" ? 1 : __kasia_mode == "공격" ? 2 : __kasia_mode == "초공격" ? 2 : 1)
weightMTF_custom = input.int(2, "HTF 합류(피벗) 가중치",   minval=0, maxval=5, group=g_weights)
weightMTF = __kasia_mode == "커스텀" ? weightMTF_custom : (__kasia_mode == "보수" ? 3 : __kasia_mode == "공격" ? 2 : __kasia_mode == "초공격" ? 1 : 2)

// 강화 기능(토글)
g_enhance = "강화 기능(토글)"
useLiqVoid     = input.bool(false, "Use Liquidity Void", group=g_enhance)
liqN           = input.int(4, "Void impulse bars (N)", minval=2, maxval=20, group=g_enhance)
liqAtrMult     = input.float(1.2, "Void ATR × threshold", step=0.1, group=g_enhance)
liqMaxAge      = input.int(30, "Void track max age (bars)", minval=5, maxval=200, group=g_enhance)

useVolDiv      = input.bool(false, "Use Volume/OBV Divergence", group=g_enhance)
divLen         = input.int(20, "Divergence lookback", minval=5, maxval=200, group=g_enhance)
divSlopeThr    = input.float(0.0, "Min |slope| to confirm div (0=off)", step=0.001, group=g_enhance)

usePatterns    = input.bool(false, "Use Candle Patterns", group=g_enhance)
patUseEngulf   = input.bool(true,  "Bull/Bear Engulfing", group=g_enhance)
patUsePin      = input.bool(false, "Pin Bar",             group=g_enhance)
pinWickRatio   = input.float(2.5,  "Pin wick/body min ratio", step=0.1, group=g_enhance)

useMTFPivot    = input.bool(false, "Use HTF Pivot Confluence", group=g_enhance)
htfPivotTf     = input.timeframe("240", "HTF for Pivot", group=g_enhance)
htfPivotLR     = input.int(5, "HTF pivot left/right", minval=2, maxval=15, group=g_enhance)
htfPivotProxATR= input.float(1.0, "Proximity ATR× (current TF)", step=0.1, group=g_enhance)

// Conflict / Arbitration
g_conflict = "신호 충돌 조정"
conflictFvgVsBBFactor = input.float(0.0, "FVG weight factor when opposite Breaker active (0~1)", minval=0.0, maxval=1.0, step=0.05, group=g_conflict)

// 확률 기반 게이트
g_prob = "확률 기반 게이트"
useProbFilter  = input.bool(true, "Enable Probability Gating", group=g_prob)
probMethod     = input.string("Logistic", "Probability method", options=["Logistic","Linear"], group=g_prob)
minProbLong    = input.float(0.66, "Min Prob Long", step=0.01, minval=0.0, maxval=1.0, group=g_prob)
minProbShort   = input.float(0.66, "Min Prob Short", step=0.01, minval=0.0, maxval=1.0, group=g_prob)
probAlpha      = input.float(1.5, "Logistic α", step=0.1, group=g_prob)
probBetaOffset = input.float(0.5, "Logistic β offset (relative to minScore)", step=0.1, group=g_prob)
redundancyK    = input.float(0.90, "BOS+CHoCH redundancy factor (multiply prob)", minval=0.5, maxval=1.0, step=0.01, group=g_prob)
ctrTrendPenalty= input.float(0.15, "Counter-trend penalty (prob × (1-pen))", minval=0.0, maxval=0.5, step=0.01, group=g_prob)
// pL/pS 사전 선언
var float pL = na
var float pS = na

// 프리셋
g_preset = "프리셋"
presetMode = input.string("사용자 지정", "프리셋", options=["사용자 지정","추세·보수","기본(밸런스)","공격 스캘프","추세·러너(트레일)","역추세·스윕"], group=g_preset)
// 익절(청산) 매니저
g_tp = "익절(청산) 매니저"
tp2Mode = input.string("RR", "TP2 모드 (롱/숏 공통)", options=["RR","NextSwing","FVG_mid","FVG_full","PrevDayHighLow","트레일만 사용"], group=g_tp)


// Effective thresholds from preset
effMinScoreLong  = presetMode=="추세·보수" ? 8 : presetMode=="기본(밸런스)" ? 7 : presetMode=="공격 스캘프" ? 6 : presetMode=="추세·러너(트레일)" ? 7 : presetMode=="역추세·스윕" ? 7 : minScoreLong
effMinScoreShort = presetMode=="추세·보수" ? 8 : presetMode=="기본(밸런스)" ? 7 : presetMode=="공격 스캘프" ? 6 : presetMode=="추세·러너(트레일)" ? 7 : presetMode=="역추세·스윕" ? 7 : minScoreShort
effMinProbLong   = presetMode=="추세·보수" ? 0.65 : presetMode=="기본(밸런스)" ? 0.60 : presetMode=="공격 스캘프" ? 0.55 : presetMode=="추세·러너(트레일)" ? 0.62 : presetMode=="역추세·스윕" ? 0.62 : minProbLong
effMinProbShort  = presetMode=="추세·보수" ? 0.65 : presetMode=="기본(밸런스)" ? 0.60 : presetMode=="공격 스캘프" ? 0.55 : presetMode=="추세·러너(트레일)" ? 0.62 : presetMode=="역추세·스윕" ? 0.62 : minProbShort
effConflictFvgVsBB= presetMode=="추세·보수" ? 0.0  : presetMode=="기본(밸런스)" ? 0.10 : presetMode=="공격 스캘프" ? 0.25 : presetMode=="추세·러너(트레일)" ? 0.15 : presetMode=="역추세·스윕" ? 0.15 : conflictFvgVsBBFactor

// ─────────────────────────────────────────────────────────────────────────────










g_fvg      = "FVG & OB Settings"
fvgLife    = input.int(15, "FVG 활성 수명(봉)", minval=1, maxval=100, group=g_fvg)
obLife     = input.int(15, "OB 활성 수명(봉)",  minval=1, maxval=100, group=g_fvg)
obAtrMult  = input.float(1.5, "OB 감지 ATR 배수",  minval=0.1, step=0.1, group=g_fvg)

g_bb       = "Breaker Block Settings"
bbLife     = input.int(15, "BB 활성 수명(봉)", minval=1, maxval=100, group=g_bb)

g_pivot    = "Pivot Settings"
pivotLen   = input.int(3, "피벗 길이", minval=2, maxval=20, group=g_pivot)
pivotProx  = input.float(0.5, "피벗 근접도(ATR)", minval=0.0, maxval=5.0, group=g_pivot)

g_mom       = "Momentum Settings"
useSqz      = input.bool(true,  "스퀴즈 모멘텀 사용", group=g_mom)
sqzBBLen    = input.int (20,   "BB 길이", minval=1, group=g_mom)
sqzBBMult   = input.float(2.0, "BB 배수",  minval=0.1, group=g_mom)
sqzKCLen    = input.int (20,   "KC 길이", minval=1, group=g_mom)
sqzKCMult   = input.float(1.5, "KC 배수",  minval=0.1, group=g_mom)
sqzUseTR    = input.bool(true,  "KC에 True Range 사용", group=g_mom)
sqzMode     = input.string("Fire", "스퀴즈 모드", options=["Compression","Fire"], group=g_mom)



// ─────────────────────────────────────────────────────────────────────────────
// [MERGE] UM Suite removed to unify filters (ADX/Session/Range/WT)


// ── ADX(진입 방지 필터) ──  — 자유 TF + 전역 precompute
g_adx        = "ADX Filter"
useAdxFilter = input.bool(true, "ADX 필터 사용", group=g_adx)
adxLen       = input.int(14, "ADX 길이(DI)", minval=1, group=g_adx)
adxSmooth    = input.int(14, "ADX 스무딩", minval=1, group=g_adx)
useAdxHTF    = input.bool(true,  "상위TF ADX 사용", group=g_adx)
adxTf        = input.timeframe("15", "ADX 상위TF", group=g_adx)
adxThreshold = input.float(23.0, "ADX 임계치", minval=1.0, group=g_adx)

// ── Dynamic Risk Controls ──
g_dynrisk = "Dynamic Risk Controls"
useDynRisk       = input.bool(true,  "동적 리스크 조절(포지션 크기 스케일링)", group=g_dynrisk)
riskScaleMin     = input.float(0.40, "최소 스케일", minval=0.1, maxval=2.0, step=0.05, group=g_dynrisk)
riskScaleMax     = input.float(1.00, "최대 스케일", minval=0.1, maxval=2.0, step=0.05, group=g_dynrisk)
useVolSize       = input.bool(true,  "변동성(ATR/가격) 높으면 사이즈↓", group=g_dynrisk)
atrLook          = input.int(14,     "ATR 룩백", minval=1, group=g_dynrisk)
volLowThr        = input.float(0.0015, "저변동 임계(ATR/Price)", step=0.0001, group=g_dynrisk)
volHighThr       = input.float(0.0050, "고변동 임계(ATR/Price)", step=0.0001, group=g_dynrisk)
useDDScale       = input.bool(true,  "자본곡선 DD 커지면 사이즈↓", group=g_dynrisk)
ddCapPct         = input.float(10.0, "DD 캡(%) → 10% DD면 스케일 저점", minval=1, step=0.5, group=g_dynrisk)
useConsLossCtl   = input.bool(true,  "연속 손실 시 사이즈↓ / 쿨다운", group=g_dynrisk)
consLossLimit    = input.int(3,      "연속 손실 기준(회)", minval=1, maxval=10, group=g_dynrisk)
consLossScale    = input.float(0.50, "연속 손실시 스케일", minval=0.1, maxval=1.0, step=0.05, group=g_dynrisk)
lossCooldownBars = input.int(30,     "연속 손실 후 쿨다운(봉)", minval=1, maxval=200, group=g_dynrisk)

// ---- Win-rate based risk adjustments ----
useWinRateAdj    = input.bool(false, "승률 기반 리스크 스케일링", group=g_dynrisk)
winRatePeriod    = input.int(20,     "승률 계산 거래수", minval=1, maxval=100, group=g_dynrisk)
winRateUpper     = input.float(0.60,  "승률 상한 (배수↑)", minval=0.0, maxval=1.0, step=0.01, group=g_dynrisk)
winRateLower     = input.float(0.40,  "승률 하한 (배수↓)", minval=0.0, maxval=1.0, step=0.01, group=g_dynrisk)
winScaleUp       = input.float(1.10,  "상한 초과 시 스케일 배수", minval=0.0, step=0.05, group=g_dynrisk)
winScaleDown     = input.float(0.90,  "하한 미만 시 스케일 배수", minval=0.0, step=0.05, group=g_dynrisk)


// ── Daily Risk Stop ── (전역에서만 호출 가능)
g_daily = "Daily Risk Stop"
useDailyLossStop   = input.bool(true,  "일일 손실 제한 사용(당일 거래 중지)", group=g_daily)
dailyLossPercent   = input.float(3.0,  "일일 최대 손실(% of equity)", minval=0.5, maxval=50, step=0.5, group=g_daily)

// ✅ 전역 호출: 비활성 시 매우 큰 값으로 우회
strategy.risk.max_intraday_loss(useDailyLossStop ? dailyLossPercent : 1000.0, strategy.percent_of_equity)// ── Exit Management ──

g_exit = "Exit Management"
useBEafterTP1  = input.bool(true,  "TP1 체결 후 BE+오프셋으로 스탑 이동", group=g_exit)
beAtrOffset    = input.float(0.20, "BE 오프셋(ATR 배)", step=0.05, group=g_exit)
useMaxBarsHold = input.bool(true,  "최대 보유봉 수 제한(시간 제한)", group=g_exit)
maxBarsInTrade = input.int(15,     "최대 보유봉 수(1m=분)", minval=1, maxval=500, group=g_exit)

g_dyn       = "Dynamic Volatility Adjustment (RR/SL)"
useDynVol   = input.bool(true, "변동성에 따라 RR/SL 동적 조정", group=g_dyn)

g_sl        = "TP/SL Settings"
rrTP1       = input.float(1.2, "TP1 R:R", minval=0.1, group=g_sl)
rrTP2       = input.float(2.0, "TP2 R:R", minval=0.1, group=g_sl)
tp1Pct      = input.float(50,  "TP1 청산 비율(%)", minval=1, maxval=99, group=g_sl)
atrSLMult   = input.float(1.0, "ATR 스탑 배수", minval=0.1, group=g_sl)
stopBasis   = input.string("Pivot", options=["Pivot", "PrevBar"], title="스탑 기준", group=g_sl)
useTP1      = input.bool(true,  "TP1 사용", group=g_sl)
useTP2      = input.bool(true,  "TP2 사용", group=g_sl)

// ── Regime-adaptive RR multipliers ──
g_rradapt   = "Regime Adaptive RR"
// When enabled, the strategy will adjust RR targets based on market regime (trend vs range)
useRegimeRR = input.bool(false, "시장 국면 적응형 RR 사용", group=g_rradapt)
trendRRMult = input.float(1.5, "추세장 TP2 배수", minval=0.1, maxval=5.0, step=0.1, group=g_rradapt)
rangeRRMult = input.float(0.8, "횡보장 TP2 배수", minval=0.1, maxval=5.0, step=0.1, group=g_rradapt)

// ─────────────────────────────────────────────────────────────────────────────
// 프리셋 강제 적용(Effective values) — defined AFTER all inputs to avoid undeclared errors
bool runnerPreset = (presetMode == "추세·러너(트레일)")

useTP1_eff         = useTP1     or runnerPreset
useTP2_eff         = useTP2 and not runnerPreset
tp1Pct_eff         = runnerPreset ? 50.0 : tp1Pct
useBEafterTP1_eff  = useBEafterTP1 or runnerPreset
trailOnlyEff       = (tp2Mode == "트레일만 사용") or runnerPreset
// ─────────────────────────────────────────────────────────────────────────────

g_fixed     = "고정 수량 옵션"
useFixedQty = input.bool(false,  "고정 수량 사용(달러 기준)", group=g_fixed)
fixedQty    = input.float(5000.0, "고정 수량($)", minval=0.0, group=g_fixed)
g_risk     = "Risk Sizing (ATR Risk %)"
useRiskPerTrade = input.bool(true,  "ATR×mult 기준 리스크% 수량", group=g_risk)
riskPct         = input.float(1.0,   "리스크 %/트레이드", minval=0.1, maxval=10.0, step=0.1, group=g_risk)


// === TF-based defaults for stops & trailing ===
is1m = timeframe.period == "1"
is5m = timeframe.period == "5"
// ATR stop and trailing multipliers by TF
atrSLMult_eff = is1m ? 1.3 : is5m ? 1.0 : atrSLMult
trailAtrMult_eff = is1m ? 1.8 : is5m ? 2.0 : 2.0
// High-vol threshold by TF (ATR/Price)
volHighThr_eff = is1m ? 0.00389 : is5m ? 0.00898 : volHighThr


g_wt       = "WaveTrend 필터"

// ── ATR Volatility Gate (added by KASIA patch) ──
g_atrvol   = "ATR 변동성 필터 (Gate)"
useATRvol  = input.bool(true,  "ATR 변동성 필터", group=g_atrvol)
atrVolLen  = input.int(14,     "ATR 길이(Vol)", 5, 200, group=g_atrvol)
atrVolEMA  = input.int(50,     "ATR 평균 길이", 5, 500, group=g_atrvol)
atrVolK    = input.float(1.02, "임계배수(ATR>EMA*배수)", 1.00, 1.20, 0.01, group=g_atrvol)
atrNow     = ta.atr(atrVolLen)
atrAvg     = ta.ema(atrNow, atrVolEMA)
volOK_ATR  = not useATRvol or (atrNow > atrAvg * atrVolK)

useWtFilter= input.bool(false,  "WaveTrend 필터 사용", group=g_wt)
wtLen      = input.int(10,      "WT ESA 길이",       minval=1, maxval=100, group=g_wt)
wtTcLen    = input.int(21,      "WT TCI 길이",       minval=1, maxval=100, group=g_wt)
wtSmooth   = input.int(4,       "WT 시그널 스무딩", minval=1, maxval=20, group=g_wt)
wtObLevel  = input.float(60.0,  "WT 과매수 레벨", group=g_wt)
wtOsLevel  = input.float(-60.0, "WT 과매도 레벨",   group=g_wt)

// ── Range & Session (HTF 자유 TF) ──
g_range     = "레인지 & 세션 필터"
useRangeFilter   = input.bool(true,  "레인지 필터 사용(HTF)", group=g_range)
rangeTf          = input.timeframe("15",  "레인지 측정 상위TF(자유)", group=g_range)
rangeBars        = input.int(20,    "레인지 측정 봉수",  minval=5, maxval=200, group=g_range)
rangePct         = input.float(0.5, "레인지 임계치(%)", minval=0.1, maxval=10.0, step=0.1, group=g_range)
useSessionFilter = input.bool(false, "세션 시간에만 거래", group=g_range)
usSession        = input.session("0930-1600", "세션(거래소 시간)", group=g_range)

// ── Heikin-Ashi 사용 여부 ──
g_heikin   = "Heikin-Ashi"
useHeikin  = input.bool(false, "추세판별에 Heikin-Ashi 사용", group=g_heikin)

// ── 스캘핑 파워업 ───
g_scalp = "Scalping Power-Ups"
useVwapFilter   = input.bool(true, "VWAP 추세 필터", group=g_scalp)
vwapSlopeLook   = input.int(3, "VWAP 기울기 룩백", minval=1, group=g_scalp)
useCooldown     = input.bool(true, "청산 후 쿨다운", group=g_scalp)
cooldownBars    = input.int(3, "쿨다운 봉수", minval=1, maxval=50, group=g_scalp)
useMinRange     = input.bool(true, "단일봉 최소 변동폭 필터", group=g_scalp)
minBarRangePct  = input.float(0.08, "최소 변동폭(%)", minval=0.01, step=0.01, group=g_scalp)
useVolSpike     = input.bool(true, "거래량 스파이크 게이트", group=g_scalp)
volLen          = input.int(20, "거래량 SMA 길이", minval=1, group=g_scalp)
volMult         = input.float(1.4, "거래량 배수", minval=1.0, step=0.1, group=g_scalp)
useVolBlock     = input.bool(false, "과도 변동성 차단(ATR/가격)", group=g_scalp)
volBlockThr     = input.float(0.008, "차단 임계(ATR/Price)", minval=0.001, step=0.0005, group=g_scalp)
showZones       = input.bool(false, "시각화 라인 표시(FVG/OB/BB)", group=g_scalp)

g_adapt = "Adaptive Score / Gates"
useAdaptiveScore = input.bool(true, "적응형 스코어/게이트 사용", group=g_adapt)
bandWidthLen  = input.int(20, "밴드폭 길이", minval=5, group=g_adapt)
bandWidthThr  = input.float(0.015, "밴드폭 스퀴즈 임계값", minval=0.001, step=0.001, group=g_adapt)
lowAdxRaise   = input.float(0.5, "약한 ADX 시 임계 가산", minval=0.0, step=0.1, group=g_adapt)
squeezeRaise  = input.float(0.4, "스퀴즈 시 임계 가산", minval=0.0, step=0.1, group=g_adapt)


// ── MTF MA Confluence (자유 TF) ──
g_ma = "MA Confluence Filter (옵션)"
useHtfMAFilter = input.bool(true, "상위TF MA 정렬 동의 필요", group=g_ma)
htfMaTF        = input.timeframe("15", "상위TF (자유 입력)", group=g_ma)
htfMaLen       = input.int(200, "상위TF MA 길이", minval=20, maxval=400, group=g_ma)

// ── Liquidity Sweep Filter (옵션) ──
g_sweep = "Liquidity Sweep Filter (옵션)"
useSweepFilter = input.bool(false, "유동성 스윕 확정 신호만 허용", group=g_sweep)
sweepLookback  = input.int(10, "스윕 레벨 룩백(봉)", minval=2, maxval=200, group=g_sweep)
sweepWickFrac  = input.float(0.40, "위크 비율(봉 범위 대비)", minval=0.05, maxval=0.9, step=0.05, group=g_sweep)

// ─── Helper ───
calc_qty(riskScale) =>
    if useFixedQty
        fixedQty * leverage / close
    else
        // Risk-based sizing (ATR × stop mult)
        // Risk-based sizing (ATR × stop mult)
        if useRiskPerTrade
            // v5-safe: compute local ATR and TF-dependent stop multiplier
            atrLocal = ta.atr(atrLook)
            tfMult   = timeframe.period == "1" ? 1.3 : timeframe.period == "5" ? 1.0 : atrSLMult
            perUnitRisk = atrLocal * tfMult
            perUnitRisk := math.max(perUnitRisk, 1e-10)
            riskDollar  = strategy.equity * (riskPct/100)
            (riskDollar * riskScale) * leverage / perUnitRisk
        else
            baseCap = strategy.equity * (strategy.opentrades == 0 ? orderPercent/100 : 0)
            (baseCap * riskScale) * leverage / close

fvgDetector() =>
    bull = (high[2] < low[1]) and (low < low[1])
    bear = (low[2] > high[1]) and (high > high[1])
    up  = bull ? low[1]  : na
    dn  = bull ? high[2] : na
    upb = bear ? low[2]  : na
    dnb = bear ? high[1] : na
    [bull, bear, bull ? up : upb, bull ? dn : dnb]

obDetector() =>
    atrVal   = ta.atr(14)
    bodySize = math.abs(close - open)
    prevBear = close[1] < open[1]
    prevBull = close[1] > open[1]
    currBull = close > open
    currBear = close < open
    bullOB    = prevBear and currBull and bodySize > atrVal * obAtrMult
    bearOB    = prevBull and currBear and bodySize > atrVal * obAtrMult
    bullUp   = bullOB ? math.max(open[1], close[1]) : na
    bullDn   = bullOB ? low[1] : na
    bearUp   = bearOB ? high[1] : na
    bearDn   = bearOB ? math.min(open[1], close[1]) : na
    [bullOB, bearOB, bullOB ? bullUp : bearUp, bullOB ? bullDn : bearDn]

// Squeeze Momentum
calcSqueeze() =>
    basisBB = ta.sma(close, sqzBBLen)
    devBB   = sqzBBMult * ta.stdev(close, sqzBBLen)
    upperBB = basisBB + devBB
    lowerBB = basisBB - devBB
    maKC    = ta.sma(close, sqzKCLen)
    tr_     = math.max(high - low, math.abs(high - close[1]), math.abs(low - close[1]))
    kcRange = sqzUseTR ? tr_ : (high - low)
    rangeMA = ta.sma(kcRange, sqzKCLen)
    upperKC = maKC + rangeMA * sqzKCMult
    lowerKC = maKC - rangeMA * sqzKCMult
    sqzOn   = (lowerBB > lowerKC) and (upperBB < upperKC)
    sqzOff  = (lowerBB < lowerKC) and (upperBB > upperKC)
    linRegSource = close - math.avg(math.avg(ta.highest(high, sqzKCLen), ta.lowest(low, sqzKCLen)), ta.sma(close, sqzKCLen))
    sqzVal = ta.linreg(linRegSource, sqzKCLen, 0)
    [sqzOn, sqzOff, sqzVal]

// HTF trend (EMA50) with optional Heikin-Ashi
htfTrend(tf) =>
    string srcTicker = useHeikin ? ticker.heikinashi(syminfo.tickerid) : syminfo.tickerid
    htfClose = request.security(srcTicker, tf, close, lookahead=barmerge.lookahead_off)
    htfEMA50 = request.security(srcTicker, tf, ta.ema(close, 50), lookahead=barmerge.lookahead_off)
    htfClose > htfEMA50 ? 1 : -1

// Pivot calc
getPivots(len) =>
    ph = ta.pivothigh(high, len, len)
    pl = ta.pivotlow (low,  len, len)
    var float lastHigh = na
    var float lastLow  = na
    if not na(ph)
        lastHigh := ph
    if not na(pl)
        lastLow  := pl
    [lastHigh, lastLow]

inProximity(priceLevel, proximityATR) =>
    atrVal = ta.atr(14)
    math.abs(close - priceLevel) <= atrVal * proximityATR

// ADX helper (DMI 기반)
adx_from_dmi(_len, _smooth) =>
    [_, _, _adx] = ta.dmi(_len, _smooth)
    _adx

// ─── States (FVG/OB/BB) ───
var bool fvgBullActive   = false
var bool fvgBullRetested = false
var float fvgBullUpper   = na
var float fvgBullLower   = na
var int   fvgBullAge     = 0
var bool fvgBearActive   = false
var bool fvgBearRetested = false
var float fvgBearUpper   = na
var float fvgBearLower   = na
var int   fvgBearAge     = 0
var bool obBullActive    = false
var bool obBullRetested  = false
var float obBullUpper    = na
var float obBullLower    = na
var int   obBullAge      = 0
var bool obBearActive    = false
var bool obBearRetested  = false
var float obBearUpper    = na
var float obBearLower    = na
var int   obBearAge      = 0
var bool bbBullActive    = false
var bool bbBullRetested  = false
var float bbBullUpper    = na
var float bbBullLower    = na
var int   bbBullAge      = 0
var bool bbBearActive    = false
var bool bbBearRetested  = false
var float bbBearUpper    = na
var float bbBearLower    = na
var int   bbBearAge      = 0

// ─── Performance state for Dynamic Risk ───
var float eqPeak           = na
var int   consLosses       = 0
var int   lastLossBar      = na
var float lastClosedEquity = na
// Track recent trade outcomes: 1 for win, 0 for loss
var float[] winHistory = array.new_float()
if na(eqPeak)
    eqPeak := strategy.equity
if na(lastClosedEquity)
    lastClosedEquity := strategy.equity

// ─── Main Logic ───

// FVG update
[detBullFVG, detBearFVG, fvgUp, fvgDn] = fvgDetector()
if detBullFVG
    fvgBullActive   := true
    fvgBullRetested := false
    fvgBullUpper    := fvgUp
    fvgBullLower    := fvgDn
    fvgBullAge      := 0
if detBearFVG
    fvgBearActive   := true
    fvgBearRetested := false
    fvgBearUpper    := fvgUp
    fvgBearLower    := fvgDn
    fvgBearAge      := 0

if fvgBullActive
    fvgBullAge += 1
    if (low <= fvgBullUpper and high >= fvgBullLower)
        fvgBullRetested := true
    if close < fvgBullLower or fvgBullAge > fvgLife
        fvgBullActive   := false
        fvgBullRetested := false
if fvgBearActive
    fvgBearAge += 1
    if (low <= fvgBearUpper and high >= fvgBearLower)
        fvgBearRetested := true
    if close > fvgBearUpper or fvgBearAge > fvgLife
        fvgBearActive   := false
        fvgBearRetested := false

// OB update + BB switch
[detBullOB, detBearOB, obUp, obDn] = obDetector()
if detBullOB
    obBullActive   := true
    obBullRetested := false
    obBullUpper    := obUp
    obBullLower    := obDn
    obBullAge      := 0
if detBearOB
    obBearActive   := true
    obBearRetested := false
    obBearUpper    := obUp
    obBearLower    := obDn
    obBearAge      := 0

if obBullActive
    obBullAge += 1
    if (low <= obBullUpper and high >= obBullLower)
        obBullRetested := true
    if close < obBullLower
        bbBearActive   := true
        bbBearRetested := false
        bbBearUpper    := obBullUpper
        bbBearLower    := obBullLower
        bbBearAge      := 0
    if close < obBullLower or obBullAge > obLife
        obBullActive   := false
        obBullRetested := false
if obBearActive
    obBearAge += 1
    if (low <= obBearUpper and high >= obBearLower)
        obBearRetested := true
    if close > obBearUpper
        bbBullActive   := true
        bbBullRetested := false
        bbBullUpper    := obBearUpper
        bbBullLower    := obBearLower
        bbBullAge      := 0
    if close > obBearUpper or obBearAge > obLife
        obBearActive   := false
        obBearRetested := false

// BB keep/expire
if bbBullActive
    bbBullAge += 1
    if (low <= bbBullUpper and high >= bbBullLower)
        bbBullRetested := true
    if close < bbBullLower or bbBullAge > bbLife
        bbBullActive   := false
        bbBullRetested := false
if bbBearActive
    bbBearAge += 1
    if (low <= bbBearUpper and high >= bbBearLower)
        bbBearRetested := true
    if close > bbBearUpper or bbBearAge > bbLife
        bbBearActive   := false
        bbBearRetested := false

// Pivot/BOS/CHoCH
[lastHigh, lastLow] = getPivots(pivotLen)
longBOS  = not na(lastHigh) and close > lastHigh
shortBOS = not na(lastLow)  and close < lastLow

float emaSource = useHeikin ? request.security(ticker.heikinashi(syminfo.tickerid), timeframe.period, close, lookahead=barmerge.lookahead_off) : close
emaFast = ta.ema(emaSource, 5)
emaSlow = ta.ema(emaSource, 21)
trendDir  = emaFast > emaSlow ? 1 : -1
chochUp   = trendDir == -1 and longBOS
chochDown = trendDir == 1  and shortBOS

// HTF trend (자유 TF)
htf1Dir = htfTrend(htf1)
htf2Dir = htfTrend(htf2)
htfBull = (htf1Dir > 0) and (htf2Dir > 0)
htfBear = (htf1Dir < 0) and (htf2Dir < 0)

// Momentum (Squeeze)
[sqzOn, sqzOff, sqzVal] = calcSqueeze()
momLong  = useSqz and sqzOn and (sqzVal > 0) and (sqzVal > nz(sqzVal[1]))
momShort = useSqz and sqzOn and (sqzVal < 0) and (sqzVal < nz(sqzVal[1]))
if not useSqz
    momLong  := true
    momShort := true

// ATR
atrVal = ta.atr(atrLook)

// ---- ADX (precompute: 현재 & 상위 TF 자유) ----
adx_cur = adx_from_dmi(adxLen, adxSmooth)
adx_htf = request.security(syminfo.tickerid, adxTf, adx_from_dmi(adxLen, adxSmooth), lookahead=barmerge.lookahead_off)
float adxVal = useAdxHTF ? adx_htf : adx_cur
// --- ADX Gate Logic (Main only)
bool adxPassMain = (not useAdxFilter) or (adxVal >= adxThreshold)
bool adx_gate_ok = adxPassMain

dynamicFactor = 1.0
if useDynVol
    dynAtrRatio = atrVal / close
    dynBBDev   = 2.0 * ta.stdev(close, 20)
    dynBBWidth = (dynBBDev * 2.0) / close
    dynMA50    = ta.sma(close, 50)
    dynMADist  = math.abs(close - dynMA50) / close
    dynMetric  = (dynAtrRatio + dynBBWidth + dynMADist) / 3.0
    dynRaw     = 1 + dynMetric
    dynamicFactor := math.max(0.5, math.min(3.0, dynRaw))

// Regime-based RR multiplier: determine if market is trending or ranging via ADX
// Default to 1.0 when disabled
rrRegimeMul = 1.0
if useRegimeRR
    rrRegimeMul := (adxVal >= adxThreshold ? trendRRMult : rangeRRMult)

// WaveTrend (LazyBear-style)
wt_ap    = (high + low + close) / 3.0
wt_esa   = ta.ema(wt_ap, wtLen)
wt_d     = ta.ema(math.abs(wt_ap - wt_esa), wtLen)
wt_ci    = (wt_ap - wt_esa) / (0.015 * wt_d)
wt1      = ta.ema(wt_ci, wtTcLen)
wt2      = ta.sma(wt1, wtSmooth)
wtLongCond  = ta.crossover(wt1, wt2) and wt1 < wtOsLevel
wtShortCond = ta.crossunder(wt1, wt2) and wt1 > wtObLevel

// HTF Range filter (자유 TF)
rangeHigh = request.security(syminfo.tickerid, rangeTf, ta.highest(high, rangeBars), lookahead=barmerge.lookahead_off)
rangeLow  = request.security(syminfo.tickerid, rangeTf, ta.lowest(low, rangeBars),  lookahead=barmerge.lookahead_off)
rangePercVal = rangeLow != 0 ? (rangeHigh - rangeLow) / rangeLow * 100.0 : 0.0
bool avoidRange = rangePercVal <= rangePct

// Session filter
bool inSession = not na(time(timeframe.period, usSession))

// MTF MA confluence gate
htfMa = request.security(syminfo.tickerid, htfMaTF, ta.ema(close, htfMaLen), lookahead=barmerge.lookahead_off)
maAgreeLong  = not useHtfMAFilter or close > htfMa
maAgreeShort = not useHtfMAFilter or close < htfMa

// Liquidity sweep gate
highestPrev = ta.highest(high, sweepLookback)[1]
lowestPrev  = ta.lowest(low,  sweepLookback)[1]
barRange    = math.max(high - low, syminfo.mintick)
upperWick   = high - math.max(open, close)
lowerWick   = math.min(open, close) - low
sweepShort  = useSweepFilter and (high > highestPrev and close < highestPrev and (upperWick / barRange >= sweepWickFrac))
sweepLong   = useSweepFilter and (low  < lowestPrev  and close > lowestPrev  and (lowerWick / barRange >= sweepWickFrac))

// Pivot proximity
nearSupport    = not na(lastLow)  and inProximity(lastLow, pivotProx)
nearResistance = not na(lastHigh) and inProximity(lastHigh, pivotProx)


// ─────────────────────────────────────────────────────────────────────────────
// Enhancement Detection + D High/Low
dHigh = request.security(syminfo.tickerid, "D", high, lookahead=barmerge.lookahead_off)[1]
dLow  = request.security(syminfo.tickerid, "D", low,  lookahead=barmerge.lookahead_off)[1]

body = math.abs(close - open)
atr14 = ta.atr(14)

// Liquidity Void detection
var bool  voidUpActive   = false
var float voidUpTop      = na
var float voidUpBottom   = na
var int   voidUpAge      = 0
var bool  voidDnActive   = false
var float voidDnTop      = na
var float voidDnBottom   = na
var int   voidDnAge      = 0

upImpulse   = useLiqVoid and ta.rising(close, liqN)  and (ta.sma(body, liqN) / atr14 > liqAtrMult)
downImpulse = useLiqVoid and ta.falling(close, liqN) and (ta.sma(body, liqN) / atr14 > liqAtrMult)

liqN_low  = ta.lowest(low,  liqN)
liqN_high = ta.highest(high, liqN)
if upImpulse
    voidUpActive := true
    voidUpTop    := liqN_high
    voidUpBottom := liqN_low
    voidUpAge    := 0
if downImpulse
    voidDnActive := true
    voidDnTop    := liqN_high
    voidDnBottom := liqN_low
    voidDnAge    := 0
if voidUpActive
    voidUpAge += 1
    if close <= voidUpBottom or voidUpAge > liqMaxAge
        voidUpActive := false
if voidDnActive
    voidDnAge += 1
    if close >= voidDnTop or voidDnAge > liqMaxAge
        voidDnActive := false

liqFavorLong  = useLiqVoid and (not voidUpActive) and (low  <= nz(voidUpBottom,  1e10))
liqFavorShort = useLiqVoid and (not voidDnActive) and (high >= nz(voidDnTop,   -1e10))
liqRiskLong   = useLiqVoid and voidUpActive
liqRiskShort  = useLiqVoid and voidDnActive

// OBV divergence
obv = ta.cum(math.sign(ta.change(close)) * nz(volume, 0))
priceSlope = ta.linreg(close, divLen, 0) - ta.linreg(close, divLen, 1)
obvSlope   = ta.linreg(obv,   divLen, 0) - ta.linreg(obv,   divLen, 1)
bullDiv = useVolDiv and (priceSlope < 0) and (obvSlope > 0)
bearDiv = useVolDiv and (priceSlope > 0) and (obvSlope < 0)

// Candle patterns
bullEngulf = usePatterns and patUseEngulf and (close > open) and (open[1] > close[1]) and (close >= open[1]) and (open <= close[1])
bearEngulf = usePatterns and patUseEngulf and (close < open) and (open[1] < close[1]) and (close <= open[1]) and (open >= close[1])

upperWick := high - math.max(close, open)
lowerWick := math.min(close, open) - low
isBullPin = usePatterns and patUsePin and (lowerWick >= pinWickRatio * body) and (upperWick <= body)
isBearPin = usePatterns and patUsePin and (upperWick >= pinWickRatio * body) and (lowerWick <= body)

// HTF Pivot proximity
htfPH = request.security(syminfo.tickerid, htfPivotTf, ta.pivothigh(high, htfPivotLR, htfPivotLR), lookahead=barmerge.lookahead_off)
htfPL = request.security(syminfo.tickerid, htfPivotTf, ta.pivotlow (low,  htfPivotLR, htfPivotLR), lookahead=barmerge.lookahead_off)
nearHTFRes = useMTFPivot and not na(htfPH) and inProximity(htfPH, htfPivotProxATR)
nearHTFSup = useMTFPivot and not na(htfPL) and inProximity(htfPL, htfPivotProxATR)

// Scoring
scoreLong  = 0.0
scoreShort = 0.0
if fvgBullActive and fvgBullRetested
    scoreLong += weightFVG
if fvgBearActive and fvgBearRetested
    scoreShort += weightFVG
if obBullActive and obBullRetested
    scoreLong += weightOB
if obBearActive and obBearRetested
    scoreShort += weightOB
if bbBullActive and bbBullRetested
    scoreLong += weightBB
if bbBearActive and bbBearRetested
    scoreShort += weightBB
if longBOS
    scoreLong += weightBOS
if shortBOS
    scoreShort += weightBOS
if chochUp
    scoreLong += weightCHoCH
if chochDown
    scoreShort += weightCHoCH
if nearSupport
    scoreLong += weightPivot
if nearResistance
    scoreShort += weightPivot
if htfBull
    scoreLong += weightTrend
if htfBear
    scoreShort += weightTrend
if momLong
    scoreLong += weightMom
if momShort
    scoreShort += weightMom

// Enhancement scoring
if liqFavorLong
    scoreLong += weightLiqVoid
if liqFavorShort
    scoreShort += weightLiqVoid
if liqRiskLong
    scoreLong -= weightLiqVoid * 0.5
if liqRiskShort
    scoreShort -= weightLiqVoid * 0.5

if bullDiv
    scoreLong += weightDiv
if bearDiv
    scoreShort += weightDiv

if bullEngulf or isBullPin or nearHTFSup
    scoreLong += weightPattern + weightMTF
if bearEngulf or isBearPin or nearHTFRes
    scoreShort += weightPattern + weightMTF

// Conflict arbitration for FVG vs opposite Breaker
if fvgBullActive and fvgBullRetested and bbBearActive
    scoreLong -= weightFVG * (1 - effConflictFvgVsBB)
if fvgBearActive and fvgBearRetested and bbBullActive
    scoreShort -= weightFVG * (1 - effConflictFvgVsBB)

// ── 임계값(실수) 분리 → 타입 충돌 방지
float thrLong  = effMinScoreLong
float thrShort = effMinScoreShort

wantLong  = scoreLong  >= thrLong
wantLong  := wantLong and volOK_ATR
wantShort = scoreShort >= thrShort
wantShort := wantShort and volOK_ATR

// Probability gating (optional)
if useProbFilter
    totalMax = weightFVG + weightOB + weightBB + weightBOS + weightCHoCH + weightPivot + weightTrend + weightMom + weightLiqVoid + weightDiv + weightPattern + weightMTF
    totalMax := totalMax <= 0 ? 1 : totalMax
    probL_raw = math.min(math.max(scoreLong  / totalMax, 0.0), 1.0)
    probS_raw = math.min(math.max(scoreShort / totalMax, 0.0), 1.0)

    if probMethod == "Logistic"
        betaL = thrLong  + probBetaOffset
        betaS = thrShort + probBetaOffset
        pL    := 1.0 / (1.0 + math.exp(-probAlpha * (scoreLong  - betaL)))
        pS    := 1.0 / (1.0 + math.exp(-probAlpha * (scoreShort - betaS)))
    else
        pL := probL_raw
        pS := probS_raw

    bothLong  = longBOS  and chochUp
    bothShort = shortBOS and chochDown
    if bothLong
        pL := nz(pL, 0.0) * redundancyK
    if bothShort
        pS := nz(pS, 0.0) * redundancyK
    if htfBear and wantLong
        pL := nz(pL, 0.0) * (1 - ctrTrendPenalty)
    if htfBull and wantShort
        pS := nz(pS, 0.0) * (1 - ctrTrendPenalty)

    wantLong  := wantLong  and (nz(pL, 0.0) >= effMinProbLong)
    wantShort := wantShort and (nz(pS, 0.0) >= effMinProbShort)

// Gates
if not adx_gate_ok
    wantLong := false
    wantShort := false

if useWtFilter
    if not wtLongCond
        wantLong := false
    if not wtShortCond
        wantShort := false

if useSqz
    if sqzMode == "Compression"
        na
    else
        if not (sqzOff and sqzVal > 0)
            wantLong := false
        if not (sqzOff and sqzVal < 0)
            wantShort := false

if useRangeFilter and avoidRange
    wantLong := false
    wantShort := false
if useSessionFilter and not inSession
    wantLong := false
    wantShort := false

// VWAP / MinRange / Volume / Cooldown
vwap = ta.vwap(hlc3)
vwapSlopeUp   = vwap > vwap[vwapSlopeLook]
vwapSlopeDown = vwap < vwap[vwapSlopeLook]
if useVwapFilter
    if not (close > vwap and vwapSlopeUp)
        wantLong := false
    if not (close < vwap and vwapSlopeDown)
        wantShort := false

barRangePct = (high - low) / math.max(low, 1e-10) * 100.0
if useMinRange and barRangePct < minBarRangePct
    wantLong := false
    wantShort := false

if useVolSpike
    vma = ta.sma(volume, volLen)
    if not (volume > vma * volMult)
        wantLong := false
        wantShort := false

var int lastExitBar = na
if strategy.position_size == 0 and strategy.position_size[1] != 0
    lastExitBar := bar_index
    // reset structural stop when position closes
    lastStructuralStop := na

// === UM Filters Gate & Adaptive Score ===
if useAdaptiveScore
    bw        = ta.stdev(close, bandWidthLen) / nz(ta.sma(close, bandWidthLen), close)
    weakTrend = not adx_gate_ok
    inSqueeze = bw < bandWidthThr
    if weakTrend
        thrLong  += lowAdxRaise
        thrShort += lowAdxRaise
    if inSqueeze
        thrLong  += squeezeRaise
        thrShort += squeezeRaise
    wantLong  := scoreLong  >= thrLong
    wantLong  := wantLong and volOK_ATR
    wantShort := scoreShort >= thrShort
    wantShort := wantShort and volOK_ATR

// [MERGE] UM filters removed; pass-through
bool passL = true
bool passS = true
cooldownActive = useCooldown and not na(lastExitBar) and (bar_index - lastExitBar <= cooldownBars)
if cooldownActive
    wantLong := false
    wantShort := false

// MTF MA gate
if not maAgreeLong
    wantLong := false
if not maAgreeShort
    wantShort := false

// Liquidity sweep gate
if useSweepFilter
    if not sweepLong
        wantLong := false
    if not sweepShort
        wantShort := false

// ─── Dynamic Risk (position size scale) ───
eqPeak := math.max(eqPeak, strategy.equity)
ddPct  = eqPeak > 0 ? (eqPeak - strategy.equity) / eqPeak * 100.0 : 0.0

if strategy.position_size == 0 and strategy.position_size[1] != 0
    // On trade close, update consecutive loss counter and win history
    tradeProfit = not na(lastClosedEquity) ? (strategy.equity - lastClosedEquity) : 0.0
    consLosses := tradeProfit < 0 ? consLosses + 1 : 0
    if tradeProfit < 0
        lastLossBar := bar_index
    // append win/loss indicator to history
    if not na(lastClosedEquity)
        winVal = tradeProfit > 0 ? 1.0 : 0.0
        array.push(winHistory, winVal)
        if array.size(winHistory) > winRatePeriod
            array.shift(winHistory)
    // update last closed equity
    lastClosedEquity := strategy.equity

volNorm = atrVal / close
volScale = 1.0
if useVolSize
    volScale := volNorm >= volHighThr_eff ? 0.5 : volNorm <= volLowThr ? 1.0 : 0.85

ddScale = useDDScale ? math.max(riskScaleMin, 1.0 - (ddPct / ddCapPct)) : 1.0
ddScale := math.min(ddScale, 1.0)

consScale = useConsLossCtl and (consLosses >= consLossLimit) ? consLossScale : 1.0

// 1. riskScaleRaw의 초기값을 먼저 계산합니다. 이 라인은 들여쓰기가 없어야 합니다.
riskScaleRaw = volScale * ddScale * consScale

// 2. if 블록 전체의 들여쓰기를 제거하여 최상위 레벨로 만듭니다.
// Apply win-rate based scaling if enabled
if useWinRateAdj
    // 3. if 블록 내부에 속한 코드들은 반드시 한 단계 들여쓰기를 유지해야 합니다.
    float winsCnt = 0.0
    int histSize = array.size(winHistory)
    // accumulate wins
    for idx = 0 to histSize - 1
        winsCnt += array.get(winHistory, idx)
    // calculate win rate (0.5 default when no history)
    float winRate = histSize > 0 ? winsCnt / histSize : 0.5
    float winScale = 1.0
    if winRate > winRateUpper
        winScale := winScaleUp
    else if winRate < winRateLower
        winScale := winScaleDown
    riskScaleRaw := riskScaleRaw * winScale

// 4. riskScale의 최종값을 계산하는 로직도 최상위 레벨에 위치해야 합니다.
riskScale = 1.0

if useDynRisk
    riskScale := math.max(riskScaleMin, math.min(riskScaleMax, riskScaleRaw))

lossCooldownActive = useConsLossCtl and (consLosses >= consLossLimit) and not na(lastLossBar) and (bar_index - lastLossBar <= lossCooldownBars)

volatilityBlock = useVolBlock and (ta.atr(atrLook) / math.max(close, 1e-10) >= volBlockThr)
tradeAllowed = not lossCooldownActive and not volatilityBlock

qty = calc_qty(tradeAllowed ? riskScale : 0.0)
// ───── Entries / Exits ─────
// 이 if 문은 최상위 레벨에 있어야 합니다. (맨 앞으로 이동)
if strategy.position_size == 0 and tradeAllowed
    // 아래의 모든 진입 로직은 위 if문이 참일 때만 실행되므로, 한 단계 들여쓰기 합니다.
    // Long
    if wantLong and not wantShort
        // 롱 진입 관련 모든 로직은 다시 한 단계 더 들여쓰기 합니다.
        entryPrice      := close
        entryBarIndex   := bar_index
        isLongTp1Hit    := false
        var float slL   = na
        if stopBasis == "Pivot" and not na(lastLow)
            slL := lastLow
        else if stopBasis == "PrevBar"
            slL := low[1]
        rr1Calc = rrTP1 * dynamicFactor
        // Apply regime multiplier for RR2 target (trend vs range)
        rr2Calc = rrTP2 * dynamicFactor * rrRegimeMul
        slMult  = ((timeframe.period == "1" ? 1.3 : timeframe.period == "5" ? 1.0 : atrSLMult)) * dynamicFactor
        if na(slL)
            slL := close - atrVal * slMult
        risk = close - slL
        if risk <= 0
            slL := close - atrVal * slMult
            risk := atrVal * slMult
        // Long targets
        tp1Val = close + risk * rr1Calc
        float tp2Val = na
        if tp2Mode == "RR"
            tp2Val := close + risk * rr2Calc
        else if tp2Mode == "NextSwing"
            tp2Val := nz(lastHigh, close + risk * rr2Calc)
        else if tp2Mode == "FVG_mid"
            tp2Val := (na(fvgBullUpper) or na(fvgBullLower)) ? (close + risk * rr2Calc) : ((fvgBullUpper + fvgBullLower) / 2.0)
        else if tp2Mode == "FVG_full"
            tp2Val := na(fvgBullUpper) ? (close + risk * rr2Calc) : fvgBullUpper
        else if tp2Mode == "PrevDayHighLow"
            tp2Val := nz(dHigh, close + risk * rr2Calc)
        tp1 := tp1Val
        tp2 := tp2Val
        // Orders
        strategy.entry("Long", strategy.long, qty=qty, comment="Long Entry")
        strategy.exit("LongTP1", from_entry="Long", limit=tp1Val, qty_percent=(useTP2_eff ? tp1Pct_eff : 100), comment="TP1")
        if useTP2_eff
            strategy.exit("LongTP2", from_entry="Long", limit=tp2Val, comment="TP2")
        slLongFrozen := slL
        // initialize structural stop with initial SL for structure-based trailing
        lastStructuralStop := slL
        strategy.exit("LongSL", from_entry="Long", stop=slL, comment="SL-init")

    // else if 문도 첫 번째 if 문과 같은 레벨로 들여쓰기 합니다.
    else if wantShort and not wantLong
        // 숏 진입 관련 모든 로직은 다시 한 단계 더 들여쓰기 합니다.
        entryPrice      := close
        entryBarIndex   := bar_index
        isShortTp1Hit   := false
        var float slS   = na
        if stopBasis == "Pivot" and not na(lastHigh)
            slS := lastHigh
        else if stopBasis == "PrevBar"
            slS := high[1]
        rr1CalcS = rrTP1 * dynamicFactor
        // Apply regime multiplier for RR2 target (trend vs range)
        rr2CalcS = rrTP2 * dynamicFactor * rrRegimeMul
        slMultS  = ((timeframe.period == "1" ? 1.3 : timeframe.period == "5" ? 1.0 : atrSLMult)) * dynamicFactor
        if na(slS)
            slS := close + atrVal * slMultS
        riskS = slS - close
        if riskS <= 0
            slS := close + atrVal * slMultS
            riskS := atrVal * slMultS
        // Short targets
        tp1sVal = close - riskS * rr1CalcS
        float tp2sVal = na
        if tp2Mode == "RR"
            tp2sVal := close - riskS * rr2CalcS
        else if tp2Mode == "NextSwing"
            tp2sVal := nz(lastLow, close - riskS * rr2CalcS)
        else if tp2Mode == "FVG_mid"
            tp2sVal := (na(fvgBearUpper) or na(fvgBearLower)) ? (close - riskS * rr2CalcS) : ((fvgBearUpper + fvgBearLower) / 2.0)
        else if tp2Mode == "FVG_full"
            tp2sVal := na(fvgBearLower) ? (close - riskS * rr2CalcS) : fvgBearLower
        else if tp2Mode == "PrevDayHighLow"
            tp2sVal := nz(dLow, close - riskS * rr2CalcS)
        tp1s := tp1sVal
        tp2s := tp2sVal
        // Orders
        strategy.entry("Short", strategy.short, qty=qty, comment="Short Entry")
        strategy.exit("ShortTP1", from_entry="Short", limit=tp1sVal, qty_percent=(useTP2_eff ? tp1Pct_eff : 100), comment="TP1")
        if useTP2_eff
            strategy.exit("ShortTP2", from_entry="Short", limit=tp2sVal, comment="TP2")
        slShortFrozen := slS
        // initialize structural stop with initial SL for structure-based trailing
        lastStructuralStop := slS
        strategy.exit("ShortSL", from_entry="Short", stop=slS, comment="SL-init")
// === UM Exit Engine (optional) ===
var int posOpenBar = na
justOpenedLong  = strategy.position_size > 0 and strategy.position_size[1] <= 0
justOpenedShort = strategy.position_size < 0 and strategy.position_size[1] >= 0
if justOpenedLong or justOpenedShort
    posOpenBar := bar_index

barsInPos = na(posOpenBar) ? 0 : (bar_index - posOpenBar)


// Remove undefined gL/gS exit guards (legacy UM engine)
//if strategy.position_size > 0 and not na(gL)
//    strategy.exit("LiqGuardL", from_entry="Long", stop=gL)
//if strategy.position_size < 0 and not na(gS)
//    strategy.exit("LiqGuardS", from_entry="Short", stop=gS)

// TP1 flags + BE stop update
// TP1 flags + BE stop update
if strategy.position_size > 0 and not isLongTp1Hit and not na(tp1)
    if high >= tp1
        isLongTp1Hit := true
        if useBEafterTP1_eff and not na(entryPrice)
            beStop = entryPrice + atrVal * beAtrOffset
            // beStop과 같은 레벨로 들여쓰기를 수정합니다.
            strategy.exit("LongSL", from_entry="Long", stop=beStop)

if strategy.position_size < 0 and not isShortTp1Hit and not na(tp1s)
    if low <= tp1s
        isShortTp1Hit := true
        if useBEafterTP1_eff and not na(entryPrice)
            beStopS = entryPrice - atrVal * beAtrOffset
            // beStopS와 같은 레벨로 들여쓰기를 수정합니다.
            strategy.exit("ShortSL", from_entry="Short", stop=beStopS)

// Breakeven market close (fallback)
if isLongTp1Hit and strategy.position_size > 0 and not na(entryPrice)
    if low <= entryPrice
        // strategy.close()를 if 블록 안으로 한 단계 들여쓰기 합니다.
        strategy.close("Long", comment="BE Close")

if isShortTp1Hit and strategy.position_size < 0 and not na(entryPrice)
    if high >= entryPrice
        // strategy.close()를 if 블록 안으로 한 단계 들여쓰기 합니다.
        strategy.close("Short", comment="BE Close")

// Signal flip closes
if strategy.position_size > 0 and wantShort and volOK_ATR
    // strategy.close()를 if 블록 안으로 한 단계 들여쓰기 합니다.
    strategy.close("Long", comment="Signal Close")

if strategy.position_size < 0 and wantLong and volOK_ATR
    // strategy.close()를 if 블록 안으로 한 단계 들여쓰기 합니다.
    strategy.close("Short", comment="Signal Close")

// Max bars in trade (time stop)
if useMaxBarsHold and strategy.position_size != 0 and not na(entryBarIndex)
    if (bar_index - entryBarIndex) >= maxBarsInTrade
        if strategy.position_size > 0
            // strategy.close()를 if 블록 안으로 한 단계 들여쓰기 합니다.
            strategy.close("Long", comment="타임스탑")
        else
            // strategy.close()를 else 블록 안으로 한 단계 들여쓰기 합니다.
            strategy.close("Short", comment="타임스탑")

// 
// ─── 스퀴즈 모멘텀 기반 청산 옵션 ───
useSmExit       = input.bool(false,  "스퀴즈 모멘텀 페이드아웃 청산 사용", group="청산 옵션")
smLen           = input.int(20,     "SM 길이", minval=5, group="청산 옵션")
smBBMult        = input.float(2.0,  "BB 배수", step=0.1, group="청산 옵션")
smKCMult        = input.float(1.5,  "KC 배수", step=0.1, group="청산 옵션")
smFadeBars      = input.int(2,      "감쇠 연속봉 수(롱/숏)", minval=1, maxval=5, group="청산 옵션")
smZeroExit      = input.bool(true,  "0선 반대 돌파 시 즉시 청산", group="청산 옵션")

// ATR Trailing (always on; supersedes native section)
var float highestInPos = na
var float lowestInPos  = na
var float trailL       = na
var float trailS       = na

longJustOpened  = strategy.position_size > 0 and strategy.position_size[1] <= 0
shortJustOpened = strategy.position_size < 0 and strategy.position_size[1] >= 0
if longJustOpened
    highestInPos := high
    trailL := slLongFrozen
if shortJustOpened
    lowestInPos := low
    trailS := slShortFrozen

// Update structural stop while in position: track recent pivots
if strategy.position_size > 0
    // raise structural stop to latest pivot low to lock in structure-based profits
    if not na(lastLow)
        if na(lastStructuralStop) or lastLow > lastStructuralStop
            lastStructuralStop := lastLow
if strategy.position_size < 0
    // lower structural stop to latest pivot high in short to lock in structure-based profits
    if not na(lastHigh)
        if na(lastStructuralStop) or lastHigh < lastStructuralStop
            lastStructuralStop := lastHigh

if strategy.position_size > 0
    highestInPos := na(highestInPos) ? high : math.max(highestInPos, high)
    // Candidate trailing stop based on ATR
    trailCandL = highestInPos - atrVal * trailAtrMult_eff
    // Breakeven after TP1 (with ATR offset)
    beStop = isLongTp1Hit and useBEafterTP1_eff and not na(entryPrice) ? entryPrice + atrVal * beAtrOffset : slLongFrozen
    // Structure-based stop: use lastStructuralStop if available
    structStopL = na(lastStructuralStop) ? -1e10 : lastStructuralStop
    newStopL = math.max(math.max(beStop, trailCandL), structStopL)
    trailL := na(trailL) ? newStopL : math.max(trailL, newStopL)
    strategy.exit("LongSL", from_entry="Long", stop=trailL)
if strategy.position_size < 0
    lowestInPos := na(lowestInPos) ? low : math.min(lowestInPos, low)
    trailCandS = lowestInPos + atrVal * trailAtrMult_eff
    beStopS = isShortTp1Hit and useBEafterTP1_eff and not na(entryPrice) ? entryPrice - atrVal * beAtrOffset : slShortFrozen
    // Structure-based stop: use lastStructuralStop if available
    structStopS = na(lastStructuralStop) ? 1e10 : lastStructuralStop
    newStopS = math.min(math.min(beStopS, trailCandS), structStopS)
    trailS := na(trailS) ? newStopS : math.min(trailS, newStopS)
    strategy.exit("ShortSL", from_entry="Short", stop=trailS)

// === Exit reason labeling (TP1/TP2/BE/Trail/SL/Flip) ===
showExitLabels = input.bool(true, "Show exit reason labels", group=g_tp)
if showExitLabels
    // Long
    longJustClosed  = strategy.position_size[1] > 0 and strategy.position_size == 0
    longReduced     = strategy.position_size[1] > strategy.position_size and strategy.position_size > 0
    // Heuristics for reasons
    hitTP1_long     = longReduced and (not na(tp1)) and high >= tp1
    hitTP2_long     = longJustClosed and (not na(tp2)) and high >= tp2
    // Stops
    _beLong         = (isLongTp1Hit and useBEafterTP1_eff and not na(entryPrice)) ? (entryPrice + atrVal * beAtrOffset) : slLongFrozen
    _trailLong      = trailL
    hitSL_long      = longJustClosed and (not na(slLongFrozen)) and (na(_trailLong) ? (low <= slLongFrozen) : (low <= math.min(slLongFrozen, _trailLong)))
    hitTrail_long   = longJustClosed and (not na(_trailLong)) and low <= _trailLong and not hitSL_long
    hitBE_long      = longJustClosed and not hitSL_long and not hitTrail_long and (low <= _beLong)
    hitFlip_long    = longJustClosed and wantShort and not wantLong
    if hitTP1_long
        label.new(bar_index, high, "TP1", style=label.style_label_up, color=color.new(color.teal, 0), textcolor=color.white)
    if hitTP2_long
        label.new(bar_index, high, "TP2", style=label.style_label_up, color=color.new(color.green, 0), textcolor=color.white)
    if hitBE_long
        label.new(bar_index, low,  "BE",  style=label.style_label_down, color=color.new(color.gray, 0), textcolor=color.black)
    if hitTrail_long
        label.new(bar_index, low,  "Trail", style=label.style_label_down, color=color.new(color.orange, 0), textcolor=color.white)
    if hitSL_long
        label.new(bar_index, low,  "SL(pivot/ATR)", style=label.style_label_down, color=color.new(color.red, 0), textcolor=color.white)
    if hitFlip_long
        label.new(bar_index, close, "Flip(반대신호)", style=label.style_label_left, color=color.new(color.purple, 0), textcolor=color.white)
    // Short
    shortJustClosed = strategy.position_size[1] < 0 and strategy.position_size == 0
    shortReduced    = strategy.position_size[1] < strategy.position_size and strategy.position_size < 0
    hitTP1_short    = shortReduced and (not na(tp1s)) and low <= tp1s
    hitTP2_short    = shortJustClosed and (not na(tp2s)) and low <= tp2s
    _beShort        = (isShortTp1Hit and useBEafterTP1_eff and not na(entryPrice)) ? (entryPrice - atrVal * beAtrOffset) : slShortFrozen
    _trailShort     = trailS
    hitSL_short     = shortJustClosed and (not na(slShortFrozen)) and (na(_trailShort) ? (high >= slShortFrozen) : (high >= math.max(slShortFrozen, _trailShort)))
    hitTrail_short  = shortJustClosed and (not na(_trailShort)) and high >= _trailShort and not hitSL_short
    hitBE_short     = shortJustClosed and not hitSL_short and not hitTrail_short and (high >= _beShort)
    hitFlip_short   = shortJustClosed and wantLong and not wantShort
    if hitTP1_short
        label.new(bar_index, low,  "TP1", style=label.style_label_down, color=color.new(color.teal, 0), textcolor=color.white)
    if hitTP2_short
        label.new(bar_index, low,  "TP2", style=label.style_label_down, color=color.new(color.green, 0), textcolor=color.white)
    if hitBE_short
        label.new(bar_index, high, "BE",  style=label.style_label_up,   color=color.new(color.gray, 0), textcolor=color.black)
    if hitTrail_short
        label.new(bar_index, high, "Trail", style=label.style_label_up, color=color.new(color.orange, 0), textcolor=color.white)
    if hitSL_short
        label.new(bar_index, high, "SL(pivot/ATR)", style=label.style_label_up, color=color.new(color.red, 0), textcolor=color.white)
    if hitFlip_short
        label.new(bar_index, close, "Flip(반대신호)", style=label.style_label_right, color=color.new(color.purple, 0), textcolor=color.white)

// ─── 스퀴즈 모멘텀 계산

sm_src = close
sm_basis = ta.sma(sm_src, smLen)
sm_dev = ta.stdev(sm_src, smLen) * smBBMult
sm_upperBB = sm_basis + sm_dev
sm_lowerBB = sm_basis - sm_dev
sm_range = ta.atr(smLen) * smKCMult
sm_upperKC = sm_basis + sm_range
sm_lowerKC = sm_basis - sm_range
sm_squeezeOn = (sm_upperBB < sm_upperKC) and (sm_lowerBB > sm_lowerKC)

// 모멘텀 값(간단형)
sm_mom = ta.linreg(sm_src - sm_basis, smLen, 0)
// 페이드아웃 청산 로직 (UM Exit OFF일 때만 직접 청산)
// 불필요한 들여쓰기를 모두 제거하여 최상위 레벨로 만듭니다.
var float smPeakL = na
var float smPeakS = na

// 롱 포지션
// if 문도 최상위 레벨에 위치해야 합니다.
if strategy.position_size > 0
    // if 블록 내부는 한 단계 들여쓰기를 유지합니다.
    smPeakL := na(smPeakL) or strategy.position_size[1] <= 0 ? sm_mom : math.max(smPeakL, sm_mom)
    // 연속 감쇠 또는 0선 하향
    fadeL = (sm_mom < sm_mom[1] and sm_mom[1] < sm_mom[2]) and sm_mom > 0
    if (fadeL and smFadeBars >= 2) or (smZeroExit and sm_mom < 0)
        strategy.close("Long", comment="SM Fade")

// 숏 포지션
// if 문도 최상위 레벨에 위치해야 합니다.
if strategy.position_size < 0
    // if 블록 내부는 한 단계 들여쓰기를 유지합니다.
    smPeakS := na(smPeakS) or strategy.position_size[1] >= 0 ? sm_mom : math.min(smPeakS, sm_mom)
    fadeS = (sm_mom > sm_mom[1] and sm_mom[1] > sm_mom[2]) and sm_mom < 0
    if (fadeS and smFadeBars >= 2) or (smZeroExit and sm_mom > 0)
        strategy.close("Short", comment="SM Fade") 
// ─── 시각화 ───
if showZones and barstate.islast
    if fvgBullActive
        line.new(bar_index - fvgBullAge + 1, fvgBullUpper, bar_index, fvgBullUpper, color=color.new(color.green, 60), width=1)
        line.new(bar_index - fvgBullAge + 1, fvgBullLower, bar_index, fvgBullLower, color=color.new(color.green, 60), width=1)
    if fvgBearActive
        line.new(bar_index - fvgBearAge + 1, fvgBearUpper, bar_index, fvgBearUpper, color=color.new(color.red, 60), width=1)
        line.new(bar_index - fvgBearAge + 1, fvgBearLower, bar_index, fvgBearLower, color=color.new(color.red, 60), width=1)
    if obBullActive
        line.new(bar_index - obBullAge + 1, obBullUpper, bar_index, obBullUpper, color=color.new(color.teal, 60), width=1)
        line.new(bar_index - obBullAge + 1, obBullLower, bar_index, obBullLower, color=color.new(color.teal, 60), width=1)
    if obBearActive
        line.new(bar_index - obBearAge + 1, obBearUpper, bar_index, obBearUpper, color=color.new(color.orange, 60), width=1)
        line.new(bar_index - obBearAge + 1, obBearLower, bar_index, obBearLower, color=color.new(color.orange, 60), width=1)
    if bbBullActive
        line.new(bar_index - bbBullAge + 1, bbBullUpper, bar_index, bbBullUpper, color=color.new(color.purple, 60), width=1)
        line.new(bar_index - bbBullAge + 1, bbBullLower, bar_index, bbBullLower, color=color.new(color.purple, 60), width=1)
    if bbBearActive
        line.new(bar_index - bbBearAge + 1, bbBearUpper, bar_index, bbBearUpper, color=color.new(color.yellow, 60), width=1)
        line.new(bar_index - bbBearAge + 1, bbBearLower, bar_index, bbBearLower, color=color.new(color.yellow, 60), width=1)

plot(lastHigh, title="Last Swing High", color=color.red, linewidth=1, style=plot.style_stepline)
plot(lastLow,  title="Last Swing Low",  color=color.green, linewidth=1, style=plot.style_stepline)

plotshape(longBOS,  title="BOS Long",  location=location.abovebar, color=color.lime,   style=shape.labelup,   text="BOS", size=size.tiny)
plotshape(shortBOS, title="BOS Short", location=location.belowbar, color=color.red,    style=shape.labeldown, text="BOS", size=size.tiny)
plotshape(chochUp,   title="CHoCH Up",  location=location.abovebar, color=color.green, style=shape.labelup,   text="CHoCH", size=size.tiny)
plotshape(chochDown, title="CHoCH Down",location=location.belowbar, color=color.orange,style=shape.labeldown, text="CHoCH", size=size.tiny)


// ============================================================================
// (KASIA module removed) — Unused protective exit and squeeze logic eliminated
// ============================================================================

// ===============================
plotchar(adx_gate_ok, title="ADX", char="▲", location=location.top, size=size.tiny, color=color.new(color.green, 0))
plotchar(useSessionFilter ? inSession : true, title="SESSION", char="S", location=location.top, size=size.tiny, color=color.new(color.blue, 0))
plotchar(useRangeFilter ? not avoidRange : true, title="RANGE", char="R", location=location.top, size=size.tiny, color=color.new(color.orange, 0))
plotchar(wantLong, title="WANT L", char="L", location=location.top, size=size.tiny, color=color.new(color.lime, 0))
plotchar(wantShort, title="WANT S", char="S", location=location.top, size=size.tiny, color=color.new(color.red, 0))
// Entry markers
plotshape(strategy.position_size[1] == 0 and strategy.position_size > 0, title="Entry Long", style=shape.triangleup, location=location.belowbar, size=size.tiny, color=color.lime)
plotshape(strategy.position_size[1] == 0 and strategy.position_size < 0, title="Entry Short", style=shape.triangledown, location=location.abovebar, size=size.tiny, color=color.red)

// TP1 / BE labels
var label tp1LabelL = na
var label tp1LabelS = na
if isLongTp1Hit and na(tp1LabelL)
    tp1LabelL := label.new(bar_index, tp1, text="TP1✓", style=label.style_label_up, color=color.new(color.lime, 70), textcolor=color.lime, size=size.tiny)
if isShortTp1Hit and na(tp1LabelS)
    tp1LabelS := label.new(bar_index, tp1s, text="TP1✓", style=label.style_label_down, color=color.new(color.red, 70), textcolor=color.red, size=size.tiny)


3. 3번 스크립트 UT봇 기반

//@version=5
// ==============================================================================
// UT BOT COMBO SCALPING STRATEGY (Reddit-Style Presets)
// Version: v1.0.2  (2025-09-08)
// Platform: TradingView Pine Script v5
// Overlay: Yes
// Author: ChatGPT (compiled from public Reddit/TV setups & open-source UT Bot docs)
// ==============================================================================
// LICENSE: For research/education. No warranty. Use at your own risk.
// ==============================================================================
// CHANGE LOG
// v1.0.2 (2025-09-08)
// - FIX: Replaced `ta.adx()` (not available) with `ta.dmi()`-based ADX getter.
// - FIX: Removed unsupported `qty_percent` arg. Now compute qty manually from equity.
// - KEEP: //@version=5 header; risk/filters/session; alerts.
// v1.0.1 (2025-09-08)
// - Added //@version=5; switched to percent sizing attempt (reverted in v1.0.2); close() per side.
// v1.0.0 (2025-09-08)
// - Initial full release.
// ==============================================================================

strategy(
     title = "UT Bot Combo Scalping Strategy (Reddit Presets) v1.0.2"
   , shorttitle = "UTBot Combo (Reddit) v1.0.2"
   , overlay = true
   , initial_capital = 10000
   , commission_type = strategy.commission.percent
   , commission_value = 0.06
   , slippage = 0
   , calc_on_order_fills = false
   , calc_on_every_tick = false
   , pyramiding = 0
   , process_orders_on_close = false
)

// ---------------------------
// INPUTS — CORE (UT Bot)
// ---------------------------
grpUT = "UT Bot Core"
ut_key      = input.float(2.0,    "Key Value (Sensitivity)", minval=0.1, step=0.1, group=grpUT)
ut_atrLen   = input.int(1,        "ATR Period", minval=1, group=grpUT)
ut_srcType  = input.string("HL2", "Price Source", options=["CLOSE","HL2","HLC3","OHLC4"], group=grpUT)
ut_useHA    = input.bool(true,    "Use Heikin Ashi for Signals (plot on normal candles)", group=grpUT)

// ---------------------------
// INPUTS — LINREG "CANDLE" SMOOTHING (OPTIONAL)
// ---------------------------
grpLR = "LinReg Smoothing (optional)"
lr_enable   = input.bool(false, "Enable LinReg smoothing (for signal source)", group=grpLR)
lr_len      = input.int(11,     "LinReg Length", minval=1, group=grpLR)
lr_smooth   = input.int(7,      "Post-smoothing (EMA)", minval=1, group=grpLR)

// ---------------------------
// INPUTS — FILTERS
// ---------------------------
grpF = "Filters"
adx_enable  = input.bool(true,  "ADX Filter ON", group=grpF)
adx_len     = input.int(14,     "ADX Length", minval=5, group=grpF)
adx_th      = input.float(15.0, "ADX Threshold", minval=1, step=0.5, group=grpF)
cooldownBars= input.int(5,      "Cooldown Bars Between Entries", minval=0, group=grpF)

// ---------------------------
// INPUTS — TIME FILTERS
// ---------------------------
grpT = "Time Filters"
use_session = input.bool(false, "Use Session Filter", group=grpT)
session     = input.session("0000-2359", "Allowed Session (Exchange Time)", group=grpT)
use_start   = input.bool(true,  "Use Start Date Filter", group=grpT)
start_year  = input.int(2022,   "Start Year", minval=1970, group=grpT)
start_month = input.int(1,      "Start Month", minval=1, maxval=12, group=grpT)
start_day   = input.int(1,      "Start Day", minval=1, maxval=31, group=grpT)

// ---------------------------
// INPUTS — RISK & EXITS
// ---------------------------
grpR = "Risk & Exits"
use_tpPerc     = input.bool(true,  "Use Fixed Take-Profit (%)", group=grpR)
tpPerc         = input.float(1.0,  "TP %", minval=0.05, step=0.05, group=grpR)
use_atrSL      = input.bool(true,  "Use ATR Stop Loss", group=grpR)
atrSL_mult     = input.float(1.2,  "ATR SL Multiplier", minval=0.1, step=0.1, group=grpR)
use_timeStop   = input.bool(true,  "Use TimeStop (bars)", group=grpR)
timeStop_bars  = input.int(6,      "TimeStop Bars", minval=1, group=grpR)
use_CEtrail    = input.bool(false, "Use Chandelier Exit Trailing (optional)", group=grpR)
CE_len         = input.int(22,     "CE Length", minval=1, group=grpR)
CE_mult        = input.float(2.5,  "CE ATR Mult", minval=0.5, step=0.1, group=grpR)

// ---------------------------
// INPUTS — DIRECTION & SIZE
// ---------------------------
grpD = "Direction & Size"
enableLongs   = input.bool(true,  "Enable Longs", group=grpD)
enableShorts  = input.bool(true,  "Enable Shorts", group=grpD)
riskFixedQty  = input.bool(false, "Use Fixed Qty (else % of equity)", group=grpD)
fixedQty      = input.float(1,    "Fixed Qty (contracts)", minval=0.0001, step=0.0001, group=grpD)
riskPerc      = input.float(10,   "Order Size as % of Equity (if Fixed OFF)", minval=0.01, maxval=100, step=0.5, group=grpD)

// ==============================================================================
// HELPERS
// ==============================================================================
var int    lastLongBar = na
var int    lastShortBar = na

get_source(_srcType) =>
    _srcType == "CLOSE"  ? close :
    _srcType == "HL2"    ? hl2 :
    _srcType == "HLC3"   ? hlc3 : ohlc4

// Heikin Ashi
haClose = (open + high + low + close) / 4.0
var float _haOpen = na
_haOpen := na(_haOpen[1]) ? (open + close)/2 : (_haOpen[1] + haClose[1]) / 2.0
haHigh = math.max(high, math.max(_haOpen, haClose))
haLow  = math.min(low,  math.min(_haOpen, haClose))

// LinReg smoothing (optional)
linregSmooth(series, len, smoothLen) =>
    base = ta.linreg(series, len, 0)
    smoothLen > 1 ? ta.ema(base, smoothLen) : base

// ADX via ta.dmi()
getADX(len) =>
    [plus, minus, adx] = ta.dmi(len, len)
    adx

// Time filters
start_ts = timestamp(year=start_year, month=start_month, day=start_day, hour=0, minute=0)
passStart = not use_start or (time >= start_ts)
passSession = not use_session or not na(time(timeframe.period, session))

// ==============================================================================
// UT BOT CORE
// ==============================================================================
srcRaw   = get_source(ut_srcType)
srcHA    = ut_useHA ? haClose : srcRaw
srcSm    = lr_enable ? linregSmooth(srcHA, lr_len, lr_smooth) : srcHA

atr      = ta.atr(ut_atrLen)
nLoss    = ut_key * atr

basis    = ut_useHA ? (haHigh + haLow)/2.0 : hl2
upBand   = basis - nLoss
dnBand   = basis + nLoss

var float trail = na
var int   dir   = 0

prevTrail = trail[1]
prevDir   = dir[1]

float calcUpper = na
float calcLower = na

calcUpper := na(prevTrail) ? upBand : (close[1] > prevTrail ? math.max(upBand, prevTrail) : upBand)
calcLower := na(prevTrail) ? dnBand : (close[1] < prevTrail ? math.min(dnBand, prevTrail) : dnBand)

newDir =
     close > calcLower ?  1 :
     close < calcUpper ? -1 : nz(prevDir, 1)

trail := newDir == 1 ? calcLower : calcUpper
dir   := newDir

buySignal  = dir == 1 and nz(prevDir) == -1 and barstate.isconfirmed
sellSignal = dir == -1 and nz(prevDir) ==  1 and barstate.isconfirmed

// Require price cross of smoothed source vs trail
crossUp    = ta.crossover(srcSm, trail)
crossDown  = ta.crossunder(srcSm, trail)

// ==============================================================================
// ENTRY CONDITIONS + FILTERS
// ==============================================================================
coolOKLong  = na(lastLongBar)  or (bar_index - lastLongBar  > cooldownBars)
coolOKShort = na(lastShortBar) or (bar_index - lastShortBar > cooldownBars)
adxVal      = getADX(adx_len)
adxOK       = adx_enable ? (adxVal >= adx_th) : true
timeOK      = passStart and passSession

longCond  = enableLongs  and timeOK and coolOKLong  and adxOK and (buySignal  and crossUp)
shortCond = enableShorts and timeOK and coolOKShort and adxOK and (sellSignal and crossDown)

// ==============================================================================
// RISK MODULE
// ==============================================================================
atr_now     = ta.atr(ut_atrLen)
long_sl     = use_atrSL ? (strategy.position_avg_price - atrSL_mult * atr_now) : na
short_sl    = use_atrSL ? (strategy.position_avg_price + atrSL_mult * atr_now) : na
long_tp     = use_tpPerc ? strategy.position_avg_price * (1 + tpPerc/100.0) : na
short_tp    = use_tpPerc ? strategy.position_avg_price * (1 - tpPerc/100.0) : na

// Chandelier Exit (optional)
highest_h   = ta.highest(high, CE_len)
lowest_l    = ta.lowest(low,  CE_len)
ce_long     = highest_h - CE_mult * atr_now
ce_short    = lowest_l  + CE_mult * atr_now

// Qty calculation (dynamic % of equity if not fixed)
qtyDyn = riskFixedQty ? fixedQty : (strategy.equity * (riskPerc/100.0)) / close

// ==============================================================================
// ORDERS
// ==============================================================================
if (longCond)
    strategy.entry(id="L", direction=strategy.long, qty=qtyDyn)
    lastLongBar := bar_index

if (shortCond)
    strategy.entry(id="S", direction=strategy.short, qty=qtyDyn)
    lastShortBar := bar_index

// Exits: fixed TP/SL + CE trail (priority: worst-case protection)
if strategy.position_size > 0
    if use_tpPerc or use_atrSL
        strategy.exit(id="L-EXIT", from_entry="L", stop=use_atrSL ? long_sl : na, limit=use_tpPerc ? long_tp : na)
    if use_CEtrail
        ceStop = math.max(ce_long, na(long_sl) ? -1e10 : long_sl)
        strategy.exit(id="L-CE", from_entry="L", stop=ceStop)

if strategy.position_size < 0
    if use_tpPerc or use_atrSL
        strategy.exit(id="S-EXIT", from_entry="S", stop=use_atrSL ? short_sl : na, limit=use_tpPerc ? short_tp : na)
    if use_CEtrail
        ceStopS = math.min(ce_short, na(short_sl) ?  1e10 : short_sl)
        strategy.exit(id="S-CE", from_entry="S", stop=ceStopS)

// TimeStop
inPosBars = ta.barssince(strategy.position_size == 0)
if use_timeStop and strategy.position_size != 0 and not na(inPosBars) and inPosBars >= timeStop_bars
    if strategy.position_size > 0
        strategy.close("L", comment="TimeStop")
    else
        strategy.close("S", comment="TimeStop")

// ==============================================================================
// PLOTTING
// ==============================================================================
plot(trail, "UT Trail", color = dir == 1 ? color.new(color.teal, 0) : color.new(color.red, 0), linewidth=2)
plotshape(buySignal and crossUp,  title="UT BUY", style=shape.triangleup,   location=location.belowbar, color=color.new(color.teal, 0), size=size.tiny, text="BUY")
plotshape(sellSignal and crossDown,title="UT SELL", style=shape.triangledown, location=location.abovebar, color=color.new(color.red, 0),  size=size.tiny, text="SELL")

// Debug ADX & session (optional)
plotchar(adx_enable ? adxVal : na, title="ADX (debug)", char="·", location=location.bottom, color=color.gray)

// ==============================================================================
// ALERTS
// ==============================================================================
alertcondition(buySignal and crossUp,   title="UT BUY",  message="UT BUY | {{ticker}} | TF={{interval}} | Close={{close}}")
alertcondition(sellSignal and crossDown,title="UT SELL", message="UT SELL | {{ticker}} | TF={{interval}} | Close={{close}}")
alertcondition(strategy.position_size > 0 and use_tpPerc and not na(long_tp)  and close >= long_tp,  title="LONG TP HIT",  message="LONG TP HIT | {{ticker}} | Close={{close}}")
alertcondition(strategy.position_size < 0 and use_tpPerc and not na(short_tp) and close <= short_tp, title="SHORT TP HIT", message="SHORT TP HIT | {{ticker}} | Close={{close}}")




4.  4번 스크립트 웨이브트렌드

// ============================================================================
// CHANGE LOG — 2025-09-14 23:23 KST (KASIA v1.5.6e — FadeOnly+OppExit)
// [Change] Opposite-signal exit 유지 (반대 시그널 발생 시 청산)
// [Remove] 임계 미도달 교차 청산(No-Threshold Opposite Cross) 전체 제거/주석 처리
// [Keep ] 기존 기능 유지(피라미딩/TP-SL/BE/가드/UI) — 임의 경량화 없음
// [Rule ] Fade 통일: SHORT→(wt1<0 & 골든크로스) 또는 (−10 상향크로스 & 골든크로스), LONG→(wt1≥0 & 데드크로스)
// ============================================================================

// ============================================================================
// CHANGE LOG — 2025-09-14 11:33 KST — KASIA Patch v1.5.6c — Naru quick prefs
// [Pref] All filters default FALSE for easier manual toggling while tuning
// [L2]   Removed separate L2 abs-threshold input; reuse WaveTrend L2 levels (osLevel2/obLevel2)
//        for pyr_L2_by_thr. Derived thresholds: thr_long_L2, thr_short_L2.
// [Keep] Full functionality preserved. No feature removal. (경량화 금지 준수)
// ============================================================================
// ===========================================================================
// CHANGE LOG — 2025-09-14 11:36 KST — KASIA Patch v1.5.6c — UI helpers & dbgTxt wrap
// [Fix][UI] f_fill_cell / f_fill_adv_cell를 스크립트 최상단(전역)으로 강제 배치 — 'function reference' 오류 방지
// [Fix][DBG] dbgTxt 문자열 결합을 str.format 기반으로 재작성 + if show_dbg 블록 안에서만 업데이트
// [Note] 'end of line without line continuation'는 대개 줄바꿈/들여쓰기 문제 → 긴 문자열을 1줄로 만들고 포맷 사용
// ===========================================================================
// === GLOBAL table helpers (전역 선언 필수) ===
var string dbgTxt = ""
f_fill_cell(_t, _c, _r, _txt, _bg) =>
    table.cell(_t, _c, _r, text=_txt, bgcolor=_bg, text_color=color.white, text_size=size.small)
f_fill_adv_cell(_t, _c, _r, _txt, _bg, _is_header=false) =>
    table.cell(_t, _c, _r, text=_txt, bgcolor=_bg, text_color=color.white, text_size=_is_header ? size.normal : size.small)

// ===========================================================================
// CHANGE LOG — 2025-09-14 11:18 KST — KASIA Patch v1.5.6b — Trailing start & dbg guard
// [Fix][EX] 모든 trailing exit에 trail_price=close 추가(즉시 트레일 시작 요구사항 반영)
// [Fix][DBG] dbgTxt 업데이트를 if show_dbg 블록 내부로 안전 이동/가드
// ===========================================================================
// ===========================================================================
// CHANGE LOG — 2025-09-14 11:05 KST — KASIA Patch v1.5.6a — Hotfix (TV errors)
// [Fix][EX] strategy.exit trailing params: trail_points → trail_offset (Pine v5 spec)
// [Fix][EX] TP/SL/Trailing block indentation normalized (700번대 라인 인덴트 붕괴 복구)
// [Fix][UI] f_fill_cell / f_fill_adv_cell 전역 재추가 (대시보드 렌더 오류 해결)
// [Fix][EX] atrVal / trailATR 계산을 조건문 밖으로 이동(일관 참조)
// [Keep]    기능 변경 없음. 경량화 금지 준수 — 전체 기능 유지.
// ===========================================================================
// ===========================================================================
// CHANGE LOG — 2025-09-13 22:55 KST — KASIA Patch v1.5.0-α — Phase 0
// [Add][VM] Volatility-Managed Sizing (일반/Downside). 로그수익 기반, EWMA 표준편차로 per-bar 타깃 변동성에 맞춰 사이즈 스로틀.
//            qty_eff=qty_now*vm_scale, L2 진입은 L2_qty*vm_scale로 적용. (dnVol ON시 scale_min↓, vol_floor↑로 보수화)
// [Add][CD] Sticky Cooldown: 쇼크 감지(oneBarMovePct≥threshold) 시 N봉 신규진입 잠금 + (옵션) 쿨다운 중 사이즈 보정.
//            거래 게이트에 shock_ok_sticky 추가.
// [Add][TR] Trailing Stop 옵션(기본 OFF): 부분익절 사용 시 TP1 50% 체결 후 잔량 50%에 트레일 적용(ATR*mult).
// [Default] 보수적 기본값: target_vol_annual=18%, vm_len=240, scale_min/max=0.2/1.6, vol_floor=1e-5, cooldown_bars=10, trail OFF.
// [Keep]    기존 기능(신호/피라미딩/TP/SL/BE/가드) 변경 없이 유지. 경량화 금지.
// ===========================================================================
// CHANGE LOG — 2025-09-13 20:15 KST — KASIA Patch v1.4.9 — Critical L1/L2 Fix
// [CRITICAL][Fix] L1 Entry Control: strategy.entry() for L1 is now executed *inside*
//                 the `if can_long_L1 / can_short_L1` blocks. Prevents unconditional entries.
// [CRITICAL][Fix] L2(abs-threshold) ID Unification: L2 absolute-threshold adds now use
//                 the same entry IDs ("WT_Long"/"WT_Short") so TP/SL/BE logic manages all size.
// [Fix]           L2(abs-threshold) requires existing position (already_long/short),
//                 respects L2_max and step filters, and increments l2_count consistently.
// [Clean]         Removed unused lastEntryBar/lastSide artifacts. 
// [Clean]         Removed duplicated engine params in strategy() call.
// [Behavior]      When `use_partial`=true, exits split 50/50 on the *total* aggregated size
//                 across L1+L2. When false, a single full exit manages the total size.
// ===========================================================================
// ===========================================================================
// CHANGE LOG — 2025-09-13 23:20 KST — KASIA Patch v1.5.1-α — Phase 0 (Hotfix+UI)
// [Fix][CD] cd_new_only 옵션 실제 반영: 신규진입(L1)과 추가진입(L2) 준비상태 분리
//            • is_ready_for_new_entry = ready_base ∧ shock_ok_sticky
//            • is_ready_for_pyramid   = ready_base ∧ (cd_new_only ? true : shock_ok_sticky)
//            • 표준 L2 및 abs-threshold L2 모두 is_ready_for_pyramid 적용
// [Add][UI] 온차트 퍼포먼스 대시보드(table): 주간/월간 PnL·승률·거래수 표시 (barstate.islast에서 계산)
// [Doc][TR] 부분익절 미사용(use_partial=false) + 트레일 ON이면 ‘진입 즉시 전량 트레일’ 동작을 주석으로 명시
// ===========================================================================
// ===========================================================================
// CHANGE LOG — 2025-09-13 23:55 KST — KASIA Patch v1.5.2-α — Phase 0 (UI fix + Advanced + Sortino W/M/Y)
// [Fix][UI] 퍼포먼스 대시보드 v1.1: 주/월 변경 시 통계 변수 리셋, 신규 거래만 누적 계산 (last_processed_trade_count)
// [Add][UI] 고급 대시보드(좌측): Profit Factor, Sortino(근사), MaxDD, Total Trades
// [Add][MT] 모니터링: 전략 Equity 기반 Sortino(주간/월간/연간, annualized) 테이블 및 디버그 라벨 표시
// [Doc]     주석/레이블 정리 및 포맷 안전화(format.mintick 사용)
// ===========================================================================
// ===========================================================================
// CHANGE LOG — 2025-09-14 00:18 KST — KASIA Patch v1.5.3-α — Phase 0 (Sortino fix, W/M only)
// [Fix][MT] Sortino 계산 교정: ta.stdev 대신 0수익(무위험 per-bar) 기준 하방편차 직접 계산
// [Perf]    주/월 윈도우에 max_lookback 적용(기본 5000) — 긴 주기에서도 성능 보호
// [UI]      통화 포맷 일관화: format.currency(\"USD\") 사용
// [Scope]   연/연율화 소티노 제거 — W/M만 표시
// ===========================================================================
// ===========================================================================
// CHANGE LOG — 2025-09-14 01:30 KST — KASIA Patch v1.5.4-α — Phase 0 (GLOBAL funcs + table init + ATR refactor)
// [Fix][UI] 함수 전역 이동: f_fill_cell / f_fill_adv_cell 전역 선언
// [Fix][UI] 테이블 생성 표준화: var table na -> if na() then table.new() (3개)
// [Fix][EX] ATR 호출 위치 교정: atrVal/trailATR를 조건문 밖에서 일관 호출
// ==========================================================================
//@version=5

strategy(
     title="WT변동플롯",
     overlay=false,
     initial_capital=10000,
     commission_type=strategy.commission.percent,
     commission_value=0.05,
     pyramiding=1,
     slippage=1,
     default_qty_type=strategy.fixed,
     calc_on_order_fills=true,
     calc_on_every_tick=true,
     process_orders_on_close=false,
     use_bar_magnifier=true)

// ─────────────────────────────────────────────────────────────────────────────
// 유틸리티 함수
// === GLOBAL table helpers (함수는 전역에서만 선언 가능) ===

// ─────────────────────────────────────────────────────────────────────────────
clamp(x, lo, hi) => math.min(math.max(x, lo), hi)

f_ema_dynamic(src, len) =>
    _l = math.max(1.0, len)
    _a = 2.0 / (_l + 1.0)
    var float _ema = na
    _ema := na(_ema[1]) ? src : _a * src + (1.0 - _a) * _ema[1]
    _ema

f_t3(src, length, vfactor) =>
    e1 = ta.ema(src,    length)
    e2 = ta.ema(e1,     length)
    e3 = ta.ema(e2,     length)
    e4 = ta.ema(e3,     length)
    e5 = ta.ema(e4,     length)
    e6 = ta.ema(e5,     length)
    c1 = -vfactor * vfactor * vfactor
    c2 =  3.0 * vfactor * vfactor + 3.0 * vfactor * vfactor * vfactor
    c3 = -6.0 * vfactor * vfactor - 3.0 * vfactor - 3.0 * vfactor * vfactor * vfactor
    c4 =  1.0 + 3.0 * vfactor + vfactor * vfactor * vfactor + 3.0 * vfactor * vfactor
    c1*e6 + c2*e5 + c3*e4 + c4*e3

f_supersmoother(src, length) =>
    _pi  = 3.141592653589793
    _rt2 = math.sqrt(2.0)
    a1   = math.exp(-(_rt2 * _pi) / math.max(length, 1))
    b1   = 2.0 * a1 * math.cos((_rt2 * _pi) / math.max(length, 1))
    c2   = b1
    c3   = -a1 * a1
    c1   = 1.0 - c2 - c3
    var float ss = na
    ss := na(ss[1]) ? src : c1 * (src + src[1]) * 0.5 + c2 * ss[1] + c3 * ss[2]
    ss

f_zlema(src, length) =>
    _len = math.max(length, 1)
    lag  = math.round((_len - 1) / 2)
    zls  = src + (src - nz(src[lag], src))
    ta.ema(zls, _len)

// Kaufman's Adaptive MA helper (no ta.sum)
f_kama(src, erLen, fastLen, slowLen) =>
    _erLen   = math.max(erLen, 1)
    fastSC   = 2.0 / (fastLen + 1.0)
    slowSC   = 2.0 / (slowLen + 1.0)
    // Change (direction) & Volatility (rolling sum of abs changes) without ta.sum()
    _chg     = math.abs(src - nz(src[_erLen], src))
    _absChg  = math.abs(ta.change(src))
    _cum     = ta.cum(_absChg)
    _vol     = _cum - nz(_cum[_erLen], 0.0)
    _er      = _vol == 0.0 ? 0.0 : _chg / _vol
    sc       = math.pow(_er * (fastSC - slowSC) + slowSC, 2.0)
    var float kama = na
    kama := na(kama[1]) ? src : kama[1] + sc * (src - kama[1])
    kama

// ─────────────────────────────────────────────────────────────────────────────
// 백테스트 기간
// ─────────────────────────────────────────────────────────────────────────────
backtestStart = input.time(timestamp("2024-01-01T00:00:00"), "백테스트 시작일", confirm=false)
inBacktest   = time >= backtestStart

// ─────────────────────────────────────────────────────────────────────────────
// 입력(Inputs)
// ─────────────────────────────────────────────────────────────────────────────
groupWT       = "WaveTrend 핵심"
groupAWT      = "Adaptive WT (상세형)"
groupWTsmooth = "WT 스무딩"
groupRisk     = "리스크 & 자본 가드"
groupTime     = "시간 필터"
groupRange    = "상위 TF 레인지 필터"
groupShock    = "충격 보호"
groupTrend    = "추세 필터"
groupStruct   = "구조 필터 (BOS)"
groupMom      = "모멘텀 필터"
groupX        = "추가 필터"
groupComb     = "결합 필터 (옵션)"
groupDiv      = "다이버전스 필터"
groupSig      = "신호 게이트"
groupAcc      = "적중률 부스터 (WT 중심)"
groupCtrl     = "거래 제어"
groupPyr      = "피라미딩 프리셋"
groupExit     = "청산 컨트롤"

// --- WaveTrend Core ---
wtLen      = input.int(10, "ESA 길이", group=groupWT, minval=1)
wtTcLen    = input.int(21, "TCI 길이", group=groupWT, minval=1)
wtSmooth   = input.int(4,  "WT2 스무딩(고정)", group=groupWT, minval=1)
obLevel1   = input.int(60,  "과매수 L1", group=groupWT)
obLevel2   = input.int(53,  "과매수 L2", group=groupWT)
osLevel1   = input.int(-60, "과매도 L1", group=groupWT)
osLevel2   = input.int(-53, "과매도 L2", group=groupWT)

// --- Adaptive WT ---
use_adaptive_wt = input.bool(false, "변동성 적응 WT 사용", group=groupAWT, tooltip="ATR% 기반으로 wtLen/wtTcLen/wtSmooth 동적 조절")
awt_vol_len     = input.int(14, "ATR 길이(변동성)", group=groupAWT, minval=1)
awt_base_ref    = input.float(2.0, "기준 변동성(%)", group=groupAWT, step=0.1, tooltip="vol_pct가 이 값보다 높으면 길이↑, 낮으면 길이↓")
awt_k_len       = input.float(0.5, "ESA 길이 계수 k_len", group=groupAWT, step=0.1)
awt_k_tci       = input.float(0.5, "TCI 길이 계수 k_tci", group=groupAWT, step=0.1)
awt_k_sm        = input.float(0.2, "WT2 스무딩 계수 k_sm",  group=groupAWT, step=0.1)
awt_len_min     = input.int(5,  "ESA 최소 길이", group=groupAWT, minval=1)
awt_len_max     = input.int(30, "ESA 최대 길이", group=groupAWT, minval=1)
awt_tci_min     = input.int(10, "TCI 최소 길이", group=groupAWT, minval=1)
awt_tci_max     = input.int(42, "TCI 최대 길이", group=groupAWT, minval=1)
awt_sm_min      = input.int(3,  "WT2 최소 SMA", group=groupAWT, minval=1)
awt_sm_max      = input.int(9,  "WT2 최대 SMA", group=groupAWT, minval=1)

// --- WT2 Smoothing Selector ---
smooth_mode = input.string("EMA", "WT2 스무딩 방식", options=["EMA","T3","SuperSmoother","ZLEMA","KAMA"], group=groupWTsmooth)
t3_v        = input.float(0.7, "T3 vfactor", minval=0.0, maxval=1.0, step=0.05, group=groupWTsmooth)
kama_er     = input.int(10,  "KAMA ER 길이", minval=1, maxval=200, group=groupWTsmooth)
kama_fast   = input.int(2,   "KAMA Fast", minval=1, maxval=50, group=groupWTsmooth)
kama_slow   = input.int(30,  "KAMA Slow", minval=2, maxval=200, group=groupWTsmooth)

// ─────────────────────────────────────────────────────────────────────────────
// WaveTrend 계산
// ─────────────────────────────────────────────────────────────────────────────
awt_vol_pct = (ta.atr(awt_vol_len) / math.max(close, 0.0000001)) * 100.0
awt_dyn_len = math.round(clamp(wtLen    + awt_k_len * (awt_vol_pct - awt_base_ref), awt_len_min, awt_len_max))
awt_dyn_tci = math.round(clamp(wtTcLen  + awt_k_tci * (awt_vol_pct - awt_base_ref), awt_tci_min, awt_tci_max))
awt_dyn_sm  = math.round(clamp(wtSmooth + awt_k_sm  * (awt_vol_pct - awt_base_ref), awt_sm_min,  awt_sm_max))

len_esa = use_adaptive_wt ? awt_dyn_len : wtLen
len_tci = use_adaptive_wt ? awt_dyn_tci : wtTcLen
len_sm  = use_adaptive_wt ? awt_dyn_sm  : wtSmooth

ap  = hlc3
esa = use_adaptive_wt ? f_ema_dynamic(ap, awt_dyn_len) : ta.ema(ap, wtLen)
d   = use_adaptive_wt ? f_ema_dynamic(math.abs(ap - esa), len_esa) : ta.ema(math.abs(ap - esa), wtLen)
ci  = (ap - esa) / (0.015 * d)
wt1 = use_adaptive_wt ? f_ema_dynamic(ci, len_tci)                 : ta.ema(ci, wtTcLen)

f_wt2_smooth(_src, _len_fixed, _len_dyn) =>
    _use_dyn = use_adaptive_wt and smooth_mode != "KAMA"
    // EMA는 동적 길이 지원, 나머지 모드(T3/SS/ZLEMA)는 고정 길이만 사용
    _wt2_ema  = _use_dyn ? f_ema_dynamic(_src, _len_dyn) : ta.ema(_src, _len_fixed)
    _wt2_t3   = f_t3(_src, _len_fixed, t3_v)
    _wt2_ss   = f_supersmoother(_src, _len_fixed)
    _wt2_zl   = f_zlema(_src, _len_fixed)
    _wt2_kama = f_kama(_src, kama_er, kama_fast, kama_slow) // KAMA는 자체 길이를 사용
    switch smooth_mode
        "EMA"            => _wt2_ema
        "T3"             => _wt2_t3
        "SuperSmoother"  => _wt2_ss
        "ZLEMA"          => _wt2_zl
        "KAMA"           => _wt2_kama
        => _wt2_ema

wt2 = f_wt2_smooth(wt1, wtSmooth, len_sm)
bullCross = ta.crossover(wt1, wt2)
bearCross = ta.crossunder(wt1, wt2)
wt_rollover = wt1 < wt1[1] and wt1[1] > wt1[2]
wt_rollup   = wt1 > wt1[1] and wt1[1] < wt1[2]

// ─────────────────────────────────────────────────────────────────────────────
// 리스크 & 자본 가드
// ─────────────────────────────────────────────────────────────────────────────
leverage           = input.float(10.0, "레버리지 (배)", group=groupRisk, minval=1.0, step=0.1)
use_fixed_qty      = input.bool(false, "고정 달러 기준 포지션 사용", group=groupRisk, tooltip="ON: 고정달러 × 레버리지 ÷ 가격\nOFF: Equity% × 레버리지 ÷ 가격")
fixed_qty_value    = input.float(10000.0, "고정 포지션 금액 ($)", group=groupRisk, minval=0.0, step=10.0)
qty_percent        = input.float(30.0, "포지션 크기 (% of Equity)", group=groupRisk, minval=0.0, maxval=1000.0, step=0.1)

calc_qty(_useFixed, _fixed_amt, _pct, _lev, _px) =>
    _useFixed ? (_fixed_amt * _lev / _px) : (strategy.equity * (_pct/100.0) * _lev / _px)
qty_now = calc_qty(use_fixed_qty, fixed_qty_value, qty_percent, leverage, close)

// ─────────────────────────────────────────────────────────────────────────────
// Phase0 — Volatility-Managed Sizing (일반 & Downside)
// ─────────────────────────────────────────────────────────────────────────────
groupVolManage = "Volatility-Managed Sizing"
use_vol_sizing     = input.bool(true,  "VM 사이징 사용", group=groupVolManage, tooltip="연율 타깃 변동성에 맞춰 포지션 사이즈를 스로틀링")
use_dnvol          = input.bool(false, "Downside-Vol만 사용", group=groupVolManage, tooltip="음수 로그수익만 사용해 하락 변동성에 더 민감하게 축소")
vm_len             = input.int(240, "VM 길이(바)", group=groupVolManage, minval=20)
target_vol_annual  = input.float(0.18, "타깃 연변동성(%)", group=groupVolManage, step=0.01)
vm_scale_min       = input.float(0.20, "스케일 하한", group=groupVolManage, step=0.01)
vm_scale_max       = input.float(1.60, "스케일 상한", group=groupVolManage, step=0.01)
vm_vol_floor       = input.float(0.00001, "Vol 하한(per-bar)", group=groupVolManage, step=0.000001)
vm_scale_min_dn    = input.float(0.10, "스케일 하한 (dnVol ON 시)", group=groupVolManage, step=0.01)
vm_vol_floor_dn    = input.float(0.00003, "Vol 하한 (dnVol ON 시)", group=groupVolManage, step=0.000001)

// per-bar returns (log)
r = math.log(math.max(close, 0.0000001) / math.max(close[1], 0.0000001))
r_used = use_dnvol ? math.min(r, 0) : r

// EWMA variance via EMA of r and r^2
vm_mu  = ta.ema(r_used, vm_len)
vm_e2  = ta.ema(r_used * r_used, vm_len)
vm_var = math.max(0.0, vm_e2 - vm_mu * vm_mu)
vol_bar = math.sqrt(vm_var)  // per-bar EWMA stdev

// bars/year from chart timeframe
_sec = timeframe.in_seconds(timeframe.period)
_tf_min = _sec > 0 ? (_sec / 60.0) : 1.0
bars_per_year = 365.0 * 24.0 * (60.0 / _tf_min)

targetVol_bar = (target_vol_annual / 100.0) / math.sqrt(math.max(bars_per_year, 1.0))
_vol_floor_eff = use_dnvol ? vm_vol_floor_dn : vm_vol_floor
_scale_min_eff = use_dnvol ? vm_scale_min_dn : vm_scale_min

_vm_scale = targetVol_bar / math.max(vol_bar, _vol_floor_eff)
vm_scale  = use_vol_sizing ? clamp(_vm_scale, _scale_min_eff, vm_scale_max) : 1.0

// Effective sizes
qty_eff = qty_now * vm_scale

kasia_binance_mmr(_lev) =>
     _lev <= 5  ? 0.004 :
     _lev <= 10 ? 0.005 :
     _lev <= 20 ? 0.006 :
     _lev <= 50 ? 0.008 : 0.010

liq_guard_enable   = input.bool(true, "레버리지 기반 청산가 가드 사용", group=groupRisk)
withdraw_enable    = input.bool(true, "순익의 일부 출금 반영", group=groupRisk)
withdraw_percent   = input.float(20.0, "출금 비율 (순익의 %)", group=groupRisk, step=0.1, minval=0.0, maxval=100.0)
min_equity_guard   = input.float(2000.0,  "최소 유효자본 (미만이면 진입 차단)", group=groupRisk, step=10.0, minval=0.0) 

netp = strategy.netprofit
effectiveEquity = withdraw_enable ? (strategy.initial_capital + netp * (1.0 - withdraw_percent/100.0)) : (strategy.initial_capital + netp)
equity_ok = effectiveEquity >= min_equity_guard

// ─────────────────────────────────────────────────────────────────────────────
// 시간 필터
// ─────────────────────────────────────────────────────────────────────────────
use_time_filter = input.bool(false, "시간 필터 사용", group=groupTime)
trade_session   = input.session("0900-2100", "거래 세션(거래소 시간대)", group=groupTime)
inSession       = not na(time(timeframe.period, trade_session))

// ─────────────────────────────────────────────────────────────────────────────
// 상위 TF 레인지 필터
// ─────────────────────────────────────────────────────────────────────────────
range_filter_enable = input.bool(false, "상위 TF 레인지 필터 사용", group=groupRange)
range_tf            = input.timeframe("60", "레인지 계산 TF", group=groupRange)
range_bars          = input.int(12, "레인지 기준 HTF 봉수", group=groupRange, minval=2)
range_threshold     = input.float(0.6, "최소 레인지 폭 % (이상일 때만 진입 허용)", group=groupRange, step=0.1)

htf_high = request.security(syminfo.tickerid, range_tf, ta.highest(high, range_bars), lookahead=barmerge.lookahead_off)[1]
htf_low  = request.security(syminfo.tickerid, range_tf, ta.lowest(low,  range_bars), lookahead=barmerge.lookahead_off)[1]
htf_cl   = request.security(syminfo.tickerid, range_tf, close, lookahead=barmerge.lookahead_off)[1]
range_pct = ((htf_high - htf_low) / math.max(htf_cl, 0.0000001)) * 100.0
range_ok  = not range_filter_enable or (range_pct >= range_threshold)

// ─────────────────────────────────────────────────────────────────────────────
// 충격 보호
// ─────────────────────────────────────────────────────────────────────────────
shock_enable    = input.bool(false,  "급격변동 차단 사용", group=groupShock)
shock_threshold = input.float(2.5,  "1봉 최대 변동 %",     group=groupShock, step=0.1, minval=0.0)
oneBarMovePct   = math.abs(close/close[1] - 1.0) * 100.0
shock_ok        = not shock_enable or (oneBarMovePct <= shock_threshold)

// ─────────────────────────────────────────────────────────────────────────────
// Phase0 — Sticky Cooldown (Shock 이후 신규진입 잠금 + 옵션 사이즈 보정)
// ─────────────────────────────────────────────────────────────────────────────
groupCooldown = "Sticky Cooldown"
use_sticky_cooldown     = input.bool(false,  "스티키 쿨다운 사용", group=groupCooldown)
cooldown_bars           = input.int(10, "쿨다운 봉수", group=groupCooldown, minval=1)
cd_new_only             = input.bool(true,  "쿨다운 중 신규진입만 금지(기존 포지션 관리는 허용)", group=groupCooldown)
cd_size_penalty_enable  = input.bool(true,  "쿨다운 중 사이즈 보정 적용", group=groupCooldown)
cd_size_penalty_factor  = input.float(0.50, "쿨다운 사이즈 보정(배수)", group=groupCooldown, step=0.05, minval=0.05, maxval=1.0)

shock_hit = shock_enable and (oneBarMovePct >= shock_threshold)
var int cooldown_until = na
if use_sticky_cooldown and shock_hit
    cooldown_until := bar_index + cooldown_bars

cooldown_active   = use_sticky_cooldown and (not na(cooldown_until)) and (bar_index <= cooldown_until)
shock_ok_sticky   = not use_sticky_cooldown or (not cooldown_active)

// 쿨다운 중 사이즈 보정
vm_scale := (cooldown_active and cd_size_penalty_enable) ? (vm_scale * cd_size_penalty_factor) : vm_scale
qty_eff  := qty_now * vm_scale

// ─────────────────────────────────────────────────────────────────────────────
// 상위 TF 추세 필터
// ─────────────────────────────────────────────────────────────────────────────
use_htf_trend_ema = input.bool(false, "상위 TF EMA 추세 사용", group=groupTrend)
htf_trend_tf      = input.timeframe("60", "EMA 추세 TF", group=groupTrend)
htf_ema_len       = input.int(50, "EMA 길이", group=groupTrend, minval=1)
ema_htf = request.security(syminfo.tickerid, htf_trend_tf, ta.ema(close, htf_ema_len), lookahead=barmerge.lookahead_off)[1]
ema_up   = close > ema_htf
ema_down = close < ema_htf

use_htf_trend_wt  = input.bool(false, "상위 TF WaveTrend 방향 사용", group=groupTrend)
htf_wt_tf         = input.timeframe("60", "WT 방향 TF", group=groupTrend)
htf_wt_use_slope  = input.bool(true, "WT1 기울기까지 요구", group=groupTrend, tooltip="롱: WT1>WT2 & WT1상승 / 숏: WT1<WT2 & WT1하락")

f_wt(_n1,_n2,_sm) =>
    _ap = hlc3, _esa = ta.ema(_ap, _n1), _d = ta.ema(math.abs(_ap - _esa), _n1), _ci = (_ap - _esa) / (0.015 * _d), _wt1 = ta.ema(_ci, _n2), _wt2 = ta.sma(_wt1, _sm), [_wt1, _wt2]
f_wt1(_n1, _n2, _sm) =>
    [__w1, __w2] = f_wt(_n1, _n2, _sm)
    __w1
f_wt2(_n1, _n2, _sm) =>
    [__w1, __w2] = f_wt(_n1, _n2, _sm)
    __w2
wt1_htf = request.security(syminfo.tickerid, htf_wt_tf, f_wt1(wtLen, wtTcLen, wtSmooth), lookahead=barmerge.lookahead_off)[1]
wt2_htf = request.security(syminfo.tickerid, htf_wt_tf, f_wt2(wtLen, wtTcLen, wtSmooth), lookahead=barmerge.lookahead_off)[1]
wt_dir_bull = wt1_htf > wt2_htf, wt_dir_bear = wt1_htf < wt2_htf, wt_slope_up = wt1_htf >= wt1_htf[1], wt_slope_dn = wt1_htf <= wt1_htf[1]
wt_ok_long  = use_htf_trend_wt ? (wt_dir_bull and (htf_wt_use_slope ? wt_slope_up : true)) : true
wt_ok_short = use_htf_trend_wt ? (wt_dir_bear and (htf_wt_use_slope ? wt_slope_dn : true)) : true
trend_ok_long  = (not use_htf_trend_ema or ema_up) and wt_ok_long
trend_ok_short = (not use_htf_trend_ema or ema_down) and wt_ok_short

// ─────────────────────────────────────────────────────────────────────────────
// 구조 필터 (BOS)
// ─────────────────────────────────────────────────────────────────────────────
use_bos_filter = input.bool(false, "BOS 필터 사용", group=groupStruct)
bos_period     = input.int(20, "BOS Lookback(봉수)", group=groupStruct, minval=1)
bullish_bos    = high > ta.highest(high[1], bos_period)
bearish_bos    = low  < ta.lowest(low[1],  bos_period)
bos_ok_long    = (not use_bos_filter) or bullish_bos
bos_ok_short   = (not use_bos_filter) or bearish_bos

// ─────────────────────────────────────────────────────────────────────────────
// 모멘텀/추가 필터
// ─────────────────────────────────────────────────────────────────────────────
use_adx = input.bool(false, "ADX 사용", group=groupMom)
di_len  = input.int(14, "DI 길이", group=groupMom, minval=1)
adx_len = input.int(14, "ADX 스무딩", group=groupMom, minval=1)
adx_th  = input.int(25, "ADX 최소 강도", group=groupMom, minval=1)

use_mfi = input.bool(false, "MFI>50/<50 사용", group=groupMom)
mfi_len = input.int(14, "MFI 길이", group=groupMom, minval=1)

use_cci = input.bool(false, "CCI>0/<0 사용", group=groupMom)
cci_len = input.int(20, "CCI 길이", group=groupMom, minval=1)

use_obv = input.bool(false, "OBV 추세 사용", group=groupMom)

[dip, dim, adx] = ta.dmi(di_len, adx_len)
trendStrong = adx >= adx_th
trendUpDI   = dip > dim
trendDnDI   = dim > dip

f_mfi(_h, _l, _c, _v, _len) =>
    _tp   = (_h + _l + _c) / 3.0
    _mf   = _tp * _v
    _posSeries = _mf * (_tp > _tp[1] ? 1.0 : 0.0)
    _negSeries = _mf * (_tp < _tp[1] ? 1.0 : 0.0)
    _pos = ta.sma(_posSeries, _len) * _len
    _neg = ta.sma(_negSeries, _len) * _len
    _mr  = _neg == 0.0 ? 0.0 : _pos / _neg
    100.0 - (100.0 / (1.0 + _mr))

mfi = f_mfi(high, low, close, volume, mfi_len)
cci = ta.cci(close, cci_len)
obv = ta.cum(close > close[1] ? volume : close < close[1] ? -volume : 0)

mfi_ok_long  = (not use_mfi) or (mfi > 50)
mfi_ok_short = (not use_mfi) or (mfi < 50)
cci_ok_long  = (not use_cci) or (cci > 0)
cci_ok_short = (not use_cci) or (cci < 0)
obv_ok_long  = (not use_obv) or (obv > obv[1])
obv_ok_short = (not use_obv) or (obv < obv[1])

adx_ok_long  = (not use_adx) or (trendStrong and trendUpDI)
adx_ok_short = (not use_adx) or (trendStrong and trendDnDI)

// Combined momentum gate (parity)
mom_ok_long  = adx_ok_long  and mfi_ok_long  and cci_ok_long  and obv_ok_long
mom_ok_short = adx_ok_short and mfi_ok_short and cci_ok_short and obv_ok_short

// ─────────────────────────────────────────────────────────────────────────────
// 추가 필터 (VWAP/Volume Spike/BB Width)
// ─────────────────────────────────────────────────────────────────────────────
use_vwap_filter = input.bool(false, "VWAP 방향 필터 사용", group=groupX)
use_vol_spike   = input.bool(false, "거래량 스파이크 필터 사용", group=groupX)
vol_ma_len      = input.int(20, "거래량 SMA 길이", group=groupX, minval=1)
use_bb_filter   = input.bool(false, "볼린저 밴드폭 필터 사용", group=groupX)
bb_len          = input.int(20, "BB 길이", group=groupX, minval=1)
bb_mult         = input.float(2.0, "BB 표준편차 배수", group=groupX, step=0.1)
bbw_min         = input.float(2.0, "최소 BB 폭 %", group=groupX, step=0.1)
bbw_max         = input.float(50.0, "최대 BB 폭 %", group=groupX, step=0.1)

vwap_ok_long  = not use_vwap_filter or (close >= ta.vwap)
vwap_ok_short = not use_vwap_filter or (close <= ta.vwap)

vol_ok = not use_vol_spike or (volume > ta.sma(volume, vol_ma_len))

[bb_mid, bb_up, bb_dn] = ta.bb(close, bb_len, bb_mult)
bbw_pct = ((bb_up - bb_dn) / math.max(bb_mid, 0.0000001)) * 100.0
bbw_ok  = not use_bb_filter or (bbw_pct >= bbw_min and bbw_pct <= bbw_max)

// ─────────────────────────────────────────────────────────────────────────────
// 결합 필터 (RSI / SuperTrend)
// ─────────────────────────────────────────────────────────────────────────────
use_rsi_filter = input.bool(false, "RSI 결합 사용", group=groupComb)
rsi_len        = input.int(14, "RSI 길이", group=groupComb, minval=1)
rsi_os         = input.int(30, "RSI 과매도", group=groupComb, minval=1, maxval=50)
rsi_ob         = input.int(70, "RSI 과매수", group=groupComb, minval=50, maxval=99)
rsi = ta.rsi(close, rsi_len)
rsi_ok_long  = (not use_rsi_filter) or (rsi <= rsi_os)
rsi_ok_short = (not use_rsi_filter) or (rsi >= rsi_ob)

use_st_filter = input.bool(false, "SuperTrend 결합 사용", group=groupComb)
st_atr_len    = input.int(10, "ST ATR 길이", group=groupComb, minval=1)
st_mult       = input.float(3.0, "ST 배수", group=groupComb, step=0.1)
[st_line, st_dir] = ta.supertrend(st_mult, st_atr_len)
st_ok_long  = (not use_st_filter) or (close >= st_line)
st_ok_short = (not use_st_filter) or (close <= st_line)

// ─────────────────────────────────────────────────────────────────────────────
// 다이버전스 필터 (토글)
// ─────────────────────────────────────────────────────────────────────────────
use_div_filter   = input.bool(false, "다이버전스 확인 사용", group=groupDiv)
div_left         = input.int(3, "피벗 Left", group=groupDiv, minval=1)
div_right        = input.int(3, "피벗 Right", group=groupDiv, minval=1)
div_mode_confirm = input.bool(false, "확인 모드(지지 다이버전스 필요)", group=groupDiv)
div_max_gap      = input.int(20, "다이버전스 허용 최대 바 간격", group=groupDiv, minval=1)

// Split explicit typed vars
var float ph1 = na
var float ph2 = na
var float pl1 = na
var float pl2 = na
var float wh1 = na
var float wh2 = na
var float wl1 = na
var float wl2 = na

var int ph1b = na
var int ph2b = na
var int pl1b = na
var int pl2b = na
var int wh1b = na
var int wh2b = na
var int wl1b = na
var int wl2b = na

pivH = ta.pivothigh(high, div_left, div_right)
pivL = ta.pivotlow(low,  div_left, div_right)
if not na(pivH)
    ph2 := ph1, ph2b := ph1b
    ph1 := pivH,  ph1b := bar_index - div_right
if not na(pivL)
    pl2 := pl1, pl2b := pl1b
    pl1 := pivL,  pl1b := bar_index - div_right

wivH = ta.pivothigh(wt1, div_left, div_right)
wivL = ta.pivotlow(wt1,  div_left, div_right)
if not na(wivH)
    wh2 := wh1, wh2b := wh1b
    wh1 := wivH,  wh1b := bar_index - div_right
if not na(wivL)
    wl2 := wl1, wl2b := wl1b
    wl1 := wivL,  wl1b := bar_index - div_right

bull_div = not na(pl1) and not na(pl2) and not na(wl1) and not na(wl2) and (pl1 < pl2) and (wl1 > wl2)
bear_div = not na(ph1) and not na(ph2) and not na(wh1) and not na(wh2) and (ph1 > ph2) and (wh1 < wh2)

div_time_ok_long  = (na(pl1b) or na(wl1b)) ? true : math.abs(pl1b - wl1b) <= div_max_gap
div_time_ok_short = (na(ph1b) or na(wh1b)) ? true : math.abs(ph1b - wh1b) <= div_max_gap

div_ok_long  = not use_div_filter or (div_mode_confirm ? (bull_div and div_time_ok_long) : (not bear_div))
div_ok_short = not use_div_filter or (div_mode_confirm ? (bear_div and div_time_ok_short) : (not bull_div))

// ─────────────────────────────────────────────────────────────────────────────
// 신호 게이트 & 적중률 부스터 (STRICT 기반)
// ─────────────────────────────────────────────────────────────────────────────
use_obos_gate = input.bool(false, "과매수/과매도 게이트 사용 (Buy<OS2, Sell>OB2)", group=groupSig)
obos_ok_long  = (not use_obos_gate) or (wt1 <= osLevel2)
obos_ok_short = (not use_obos_gate) or (wt1 >= obLevel2)

use_zero_gate   = input.bool(false,  "제로라인 위치 필터 (롱: WT1<0 / 숏: WT1>0)", group=groupAcc)
use_reset_gate2 = input.bool(false,  "OB/OS 리셋 후 첫 크로스만 사용",               group=groupAcc, tooltip="롱: 최근 WT1<=OS2 발생 후 교차 / 숏: 최근 WT1>=OB2 발생 후 교차")
reset_lookback2 = input.int(60,     "리셋 유효 바수", group=groupAcc, minval=1)
use_min_spread  = input.bool(false,  "최소 WT 스프레드 요구", group=groupAcc)
min_spread_pts  = input.float(3.0,  "스프레드 최소값(포인트)", group=groupAcc, step=0.1)
use_min_slope   = input.bool(false,  "WT1 기울기 최소 요구", group=groupAcc)
min_slope_val   = input.float(0.3,  "WT1 최소 기울기(포인트/바)", group=groupAcc, step=0.1)

zero_ok_long  = (not use_zero_gate)  or (wt1 < 0)
zero_ok_short = (not use_zero_gate)  or (wt1 > 0)

barsSinceOS2 = nz(ta.barssince(wt1 <= osLevel2))
barsSinceOB2 = nz(ta.barssince(wt1 >= obLevel2))
reset_ok_long  = (not use_reset_gate2) or (barsSinceOS2 <= reset_lookback2)
reset_ok_short = (not use_reset_gate2) or (barsSinceOB2 <= reset_lookback2)

spread_ok_long  = (not use_min_spread) or (math.abs(wt1 - wt2) >= min_spread_pts)
spread_ok_short = (not use_min_spread) or (math.abs(wt1 - wt2) >= min_spread_pts)

slope1 = wt1 - wt1[1]
slope2 = wt2 - wt2[1]
slope_ok_long  = (not use_min_slope) or ((slope1 >= min_slope_val) and (slope2 >= 0))
slope_ok_short = (not use_min_slope) or ((slope1 <= -min_slope_val) and (slope2 <= 0))

acc_ok_long  = zero_ok_long  and reset_ok_long  and spread_ok_long  and slope_ok_long
acc_ok_short = zero_ok_short and reset_ok_short and spread_ok_short and slope_ok_short

// STRICT 진입 조건 (mom_ok_* 적용)
longCondition_core  = bullCross
shortCondition_core = bearCross

longCondition_strict  = (longCondition_core  and acc_ok_long  and trend_ok_long  and bos_ok_long  and obos_ok_long  and vwap_ok_long  and vol_ok and bbw_ok and div_ok_long  and mom_ok_long)  and rsi_ok_long and st_ok_long
shortCondition_strict = (shortCondition_core and acc_ok_short and trend_ok_short and bos_ok_short and obos_ok_short and vwap_ok_short and vol_ok and bbw_ok and div_ok_short and mom_ok_short) and rsi_ok_short and st_ok_short

// ─────────────────────────────────────────────────────────────────────────────
// 글로벌 거래 컨트롤
// ─────────────────────────────────────────────────────────────────────────────
reentry_delay_bars = input.int(0, "재진입 지연(봉)", group=groupCtrl, minval=0)
allow_reverse      = input.bool(true, "반대 시그널 즉시 전환 허용", group=groupCtrl)

var int lastFlatBar = na
posClosed = (strategy.position_size == 0 and strategy.position_size[1] != 0)
if posClosed
    lastFlatBar := bar_index
reentry_ok = allow_reverse ? true : (na(lastFlatBar) or (bar_index - lastFlatBar >= reentry_delay_bars))

// ─────────────────────────────────────────────────────────────────────────────
// 임계값 해석 (L1/L2)
// ─────────────────────────────────────────────────────────────────────────────
obShallow = math.min(obLevel1, obLevel2)
obDeep    = math.max(obLevel1, obLevel2)
osShallow = math.max(osLevel1, osLevel2)
osDeep    = math.min(osLevel1, osLevel2)

within_L1_long  = wt1 <= osShallow
within_L2_long  = wt1 <= osDeep
within_L1_short = wt1 >= obShallow
within_L2_short = wt1 >= obDeep

// ─────────────────────────────────────────────────────────────────────────────
// 피라미딩 프리셋
// ─────────────────────────────────────────────────────────────────────────────
L2_step_pct     = input.float(0.0, "L2 추가진입 간격(평단 대비 %)", group=groupPyr, step=0.1)
L2_add_qty_pct  = input.float(100.0, "L2 추가 수량 비율(%)", group=groupPyr, step=1.0)
L2_max          = input.int(1, "L2 최대 횟수", group=groupPyr, minval=0, maxval=1)
var int l2_count = 0
if strategy.position_size[1] == 0 and strategy.position_size != 0
    l2_count := 0
if strategy.position_size == 0
    l2_count := 0

L2_qty = qty_now * (L2_add_qty_pct/100.0)
long_step_ok  = (L2_step_pct <= 0.0) or (strategy.position_size <= 0 ? true : close <= strategy.position_avg_price * (1.0 - L2_step_pct/100.0))
short_step_ok = (L2_step_pct <= 0.0) or (strategy.position_size >= 0 ? true : close >= strategy.position_avg_price * (1.0 + L2_step_pct/100.0))

// ─────────────────────────────────────────────────────────────────────────────
// 진입 로직 — STRICT + 임계값 + 피라미딩(2회)
// ─────────────────────────────────────────────────────────────────────────────
// 준비 상태 (신규/L2 분리)
ready_base = inBacktest and equity_ok and range_ok and shock_ok and reentry_ok and (not use_time_filter or inSession)
is_ready_for_new_entry = ready_base and shock_ok_sticky
is_ready_for_pyramid   = ready_base and (cd_new_only ? true : shock_ok_sticky)
can_long_L1  = is_ready_for_new_entry and longCondition_strict  and within_L1_long  and (allow_reverse ? strategy.position_size <= 0 : strategy.position_size == 0)
can_short_L1 = is_ready_for_new_entry and shortCondition_strict and within_L1_short and (allow_reverse ? strategy.position_size >= 0 : strategy.position_size == 0)
can_long_L2  = is_ready_for_pyramid and longCondition_strict  and within_L2_long  and (strategy.position_size > 0) and (l2_count < L2_max) and long_step_ok
can_short_L2 = is_ready_for_pyramid and shortCondition_strict and within_L2_short and (strategy.position_size < 0) and (l2_count < L2_max) and short_step_ok

// --- L1/L2 표준 진입 (조건부 블록 내부에서만 실행) ---
if can_long_L1
    strategy.entry("WT_Long", strategy.long, qty=qty_eff, comment="WT L1")
if can_long_L2
    strategy.entry("WT_Long", strategy.long, qty=L2_qty*vm_scale, comment="WT L2")
    l2_count += 1
if can_short_L1
    strategy.entry("WT_Short", strategy.short, qty=qty_eff, comment="WT S1")
if can_short_L2
    strategy.entry("WT_Short", strategy.short, qty=L2_qty*vm_scale, comment="WT S2")
    l2_count += 1

// ─────────────────────────────────────────────────────────────
// Pyramiding — L2 by Absolute Threshold at Cross (동일 ID 사용)
// ─────────────────────────────────────────────────────────────
groupPyrAbs   = "Pyramiding — L2 by |WT| Threshold"
pyr_L2_by_thr = input.bool(true,  "L2 피라미딩: 절대 임계에서의 교차", group=groupPyrAbs, tooltip="|WT1| ≥ 임계에서 골든/데드 교차 발생 시 L2 추가 진입 (기존 ID 사용)")

// Use WT L2 levels as absolute thresholds for L2-by-threshold
thr_long_L2  = math.abs(osLevel2)
thr_short_L2 = math.abs(obLevel2)

golden_wt = ta.crossover(wt1, wt2)
dead_wt   = ta.crossunder(wt1, wt2)

already_long  = strategy.position_size > 0
already_short = strategy.position_size < 0

l2_long_abs  = pyr_L2_by_thr and is_ready_for_pyramid and already_long  and (l2_count < L2_max) and long_step_ok  and golden_wt and (math.abs(wt1) >= thr_long_L2) and (wt1 < 0)
l2_short_abs = pyr_L2_by_thr and is_ready_for_pyramid and already_short and (l2_count < L2_max) and short_step_ok and dead_wt   and (math.abs(wt1) >= thr_short_L2) and (wt1 > 0)

if l2_long_abs
    strategy.entry("WT_Long",  strategy.long,  qty=L2_qty*vm_scale, comment="WT L2(abs-thr)")
    l2_count += 1
if l2_short_abs
    strategy.entry("WT_Short", strategy.short, qty=L2_qty*vm_scale, comment="WT S2(abs-thr)")
    l2_count += 1

// ─────────────────────────────────────────────────────────────────────────────
// 청산 로직 — 임계 미도달 교차 + WT 롤오버/롤업 + TP/SL/부분익절/BE + 레버리지 가드
// ─────────────────────────────────────────────────────────────────────────────
// [v1.5.6e] REMOVED: exit_long_no_thresh  = (strategy.position_size > 0) and bearCross and (not within_L1_short)
// // [v1.5.6e] REMOVED: exit_short_no_thresh = (strategy.position_size < 0) and bullCross and (not within_L1_long)
// // if exit_long_no_thresh
// //     strategy.close_all(comment="Fade Exit (No-Threshold) LONG")
// // if exit_short_no_thresh
// //     strategy.close_all(comment="Fade Exit (No-Threshold) SHORT")
// // 
// // use_tp_sl_exits      = input.bool(true, "TP/SL 기반 익절·손절 사용", group=groupExit, tooltip="OFF: 신호 기반(교차/페이드/임계 미도달) 청산만 사용")
// // use_partial          = input.bool(true, "부분 익절 50/50", group=groupExit)
// // use_be_after_tp1     = input.bool(true, "TP1 후 손절가를 본절로", group=groupExit)
// // opposite_exit_enable = input.bool(true, "반대 WT 교차 시 청산(스위칭 OFF일 때 권장)", group=groupExit)
// // fade_exit_enable     = input.bool(true, "WT 꺾임 페이드 청산", group=groupExit)
// // use_atr_sl_tp        = input.bool(false, "ATR 기반 SL/TP 사용", group=groupExit)
// // atr_len              = input.int(14, "ATR 길이", group=groupExit, minval=1)
// // atr_mult_sl          = input.float(1.5, "ATR SL 배수", group=groupExit, step=0.1)
// // atr_rr               = input.float(2.0, "리스크/보상 (TP2용)", group=groupExit, step=0.1)
// // tp_perc              = input.float(2.0, "TP % (총 목표, ATR OFF 시)", group=groupExit, step=0.1, minval=0)
// // sl_perc              = input.float(1.0, "SL % (ATR OFF 시)",         group=groupExit, step=0.1, minval=0)
// // 
// // // Trailing 옵션 (기본 OFF)
// // use_trailing_exit   = input.bool(false, "트레일링 스탑(잔량) 사용", group=groupExit, tooltip="부분익절 사용 시 TP1 체결 후 잔량(50%)을 ATR*mult 기준으로 트레일")
// // trail_atr_len       = input.int(14, "트레일 ATR 길이", group=groupExit, minval=1)
// // trail_mult          = input.float(1.5, "트레일 ATR 배수", group=groupExit, step=0.1)
// // 
// // // (공통) ATR 사전 계산 — 조건문 밖에서 일관 호출
// // atrVal = ta.atr(atr_len)
// // trailATR = ta.atr(trail_atr_len)
// // 
// // if use_tp_sl_exits and strategy.position_size != 0
// //     // --- 평균 진입가 ---
// //     avg_price = strategy.position_avg_price
// // 
// //     // --- SL/TP 가격 계산 (ATR 또는 퍼센트)
// //     float long_sl_price  = na
// //     float long_tp_price  = na
// //     float short_sl_price = na
// //     float short_tp_price = na
// // 
// //     if use_atr_sl_tp
// //         long_sl_price  := avg_price - atrVal * atr_mult_sl
// //         long_tp_price  := avg_price + atrVal * atr_mult_sl * atr_rr
// //         short_sl_price := avg_price + atrVal * atr_mult_sl
// //         short_tp_price := avg_price - atrVal * atr_mult_sl * atr_rr
// //     else
// //         long_sl_price  := avg_price * (1 - sl_perc / 100)
// //         long_tp_price  := avg_price * (1 + tp_perc / 100)
// //         short_sl_price := avg_price * (1 + sl_perc / 100)
// //         short_tp_price := avg_price * (1 - tp_perc / 100)
// // 
// //     // --- TP1(부분익절 중간가) ---
// //     long_tp1_price  = use_partial ? avg_price + (long_tp_price  - avg_price) / 2 : long_tp_price
// //     short_tp1_price = use_partial ? avg_price - (avg_price - short_tp_price) / 2 : short_tp_price
// // 
// //     // --- TP1 체결 여부 추적 ---
// //     var bool tp1_done = false
// //     if strategy.position_size[1] == 0 and strategy.position_size != 0
// //         tp1_done := false
// //     if strategy.position_size == 0
// //         tp1_done := false
// //     if use_partial and not tp1_done
// //         if strategy.position_size > 0 and close >= long_tp1_price
// //             tp1_done := true
// //         if strategy.position_size < 0 and close <= short_tp1_price
// //             tp1_done := true
// // 
// //     // --- 본절/SL 전환가 ---
// //     long_be_price  = use_be_after_tp1 and tp1_done ? avg_price : long_sl_price
// //     short_be_price = use_be_after_tp1 and tp1_done ? avg_price : short_sl_price
// // 
// //     // --- 청산 실행 ---
// //     if use_partial
// //         // TP1: 50%
// //         strategy.exit("L_TP1_Exit", from_entry="WT_Long",  qty_percent=50, limit=long_tp1_price,  stop=long_sl_price)
// //         strategy.exit("S_TP1_Exit", from_entry="WT_Short", qty_percent=50, limit=short_tp1_price, stop=short_sl_price)
// // 
// //         // 잔량 50%: 트레일 또는 BE/TP2
// //         if use_trailing_exit
// //             t_off = trailATR * trail_mult
// //             strategy.exit("L_Trail_Exit", from_entry="WT_Long",  qty_percent=50, trail_offset=t_off, trail_price=close)
// //             strategy.exit("S_Trail_Exit", from_entry="WT_Short", qty_percent=50, trail_offset=t_off, trail_price=close)
// //         else
// //             strategy.exit("L_BE_Exit",  from_entry="WT_Long",  qty_percent=50, limit=long_tp_price,  stop=long_be_price)
// //             strategy.exit("S_BE_Exit",  from_entry="WT_Short", qty_percent=50, limit=short_tp_price, stop=short_be_price)
// //     else
// //         // 전량 한 번에
// //         if use_trailing_exit
// //             t_off = trailATR * trail_mult
// //             strategy.exit("L_Full_Trail", from_entry="WT_Long",  trail_offset=t_off, trail_price=close)
// //             strategy.exit("S_Full_Trail", from_entry="WT_Short", trail_offset=t_off, trail_price=close)
// //         else
// //             strategy.exit("L_Full_Exit", from_entry="WT_Long",  limit=long_tp_price,  stop=long_sl_price)
// //             strategy.exit("S_Full_Exit", from_entry="WT_Short", limit=short_tp_price, stop=short_sl_price)
// // 
// // if (not allow_reverse) and opposite_exit_enable
// //     if bearCross and (not within_L1_short) and strategy.position_size > 0
// //         strategy.close("WT_Long", comment="Opposite WT")
// //     if bullCross and (not within_L1_long) and strategy.position_size < 0
// //         strategy.close("WT_Short", comment="Opposite WT")
// // 
// // if fade_exit_enable
// //     if wt_rollover and strategy.position_size > 0
// //         strategy.close("WT_Long", comment="WT Fade Exit")
// //     if wt_rollup and strategy.position_size < 0
// //         strategy.close("WT_Short", comment="WT Fade Exit")
// // 
// // // ─────────────────────────────────────────────────────────────────────────────
// // // 레버리지 기반 청산가 가드
// // // ─────────────────────────────────────────────────────────────────────────────
// // if liq_guard_enable and strategy.position_size != 0
// //     mmr = kasia_binance_mmr(leverage)
// //     fee = 0.0006
// //     lossFrac = 1.0/leverage + mmr + fee
// //     guard_long_price  = strategy.position_avg_price * (1.0 - lossFrac)
// //     guard_short_price = strategy.position_avg_price * (1.0 + lossFrac)
// //     if strategy.position_size > 0 and close <= guard_long_price
// //         strategy.close_all(comment="Liq Guard (Long)")
// //     if strategy.position_size < 0 and close >= guard_short_price
// //         strategy.close_all(comment="Liq Guard (Short)")
// // 
// // // ─────────────────────────────────────────────────────────────────────────────
// // // 시각화
// // // ─────────────────────────────────────────────────────────────────────────────
// // plot(0, color=color.new(color.gray, 70), title="Zero")
// // plot(obLevel1, color=color.new(color.red,   60), title="OB L1")
// // plot(obLevel2, color=color.new(color.red,   80), title="OB L2")
// // plot(osLevel1, color=color.new(color.teal,  60), title="OS L1")
// // plot(osLevel2, color=color.new(color.teal,  80), title="OS L2")
// // 
// // plot(wt1, color=color.new(color.lime,   0), title="WT1")
// // plot(wt2, color=color.new(color.orange, 0), title="WT2")
// // plot(wt1 - wt2, color=color.new(color.blue, 85), style=plot.style_area, title="WT Spread")
// // 
// // plotshape(bullCross and within_L1_long,  title="Bull@L1", style=shape.triangleup,   location=location.bottom, size=size.tiny, color=color.new(color.teal, 0))
// // plotshape(bullCross and within_L2_long,  title="Bull@L2", style=shape.triangleup,   location=location.bottom, size=size.tiny, color=color.new(color.aqua, 0))
// // plotshape(bearCross and within_L1_short, title="Bear@L1", style=shape.triangledown, location=location.top,    size=size.tiny, color=color.new(color.red,  0))
// // plotshape(bearCross and within_L2_short, title="Bear@L2", style=shape.triangledown, location=location.top,    size=size.tiny, color=color.new(color.maroon, 0))
// // plotshape((bullCross and not within_L1_long) and (wt1 < 0),  title="Bull(NoThresh)", style=shape.circle, location=location.top,    size=size.tiny, color=color.new(color.gray, 0))
// // plotshape((bearCross and not within_L1_short) and (wt1 > 0), title="Bear(NoThresh)", style=shape.circle, location=location.bottom, size=size.tiny, color=color.new(color.gray, 0))
// // 
// // // 경고: 신규 abs-L2 조건도 포함하여 알림
// // alertcondition(can_long_L1 or can_long_L2 or l2_long_abs,  title="WT_Long_Alert",  message="WT LONG {{ticker}} TF:{{interval}}")
// // alertcondition(can_short_L1 or can_short_L2 or l2_short_abs, title="WT_Short_Alert", message="WT SHORT {{ticker}} TF:{{interval}}")
// // 
// // show_dbg = input.bool(false, "디버그 패널 표시", group="디버그")
// // var label dbg = na
// // if show_dbg
// //     dbgTxt = "qty="+str.tostring(qty_now, format.mintick)+" vm="+str.tostring(vm_scale, format.mintick)+" vol="+str.tostring(vol_bar, format.mintick)+" tgtBar="+str.tostring(targetVol_bar, format.mintick)+" dnVol="+str.tostring(use_dnvol)+ " cd="+str.tostring(cooldown_active)+ "\nopentrades="+str.tostring(strategy.opentrades) + "\nwt1="+str.tostring(wt1, format.mintick)+" wt2="+str.tostring(wt2, format.mintick) + "\nL1(L/S)="+str.tostring(within_L1_long)+"/"+str.tostring(within_L1_short)+"  L2(L/S)="+str.tostring(within_L2_long)+"/"+str.tostring(within_L2_short) + "\ntrend_ok(L/S)="+str.tostring(trend_ok_long)+"/"+str.tostring(trend_ok_short)+" range_ok="+str.tostring(range_ok)+" shock_ok="+str.tostring(shock_ok)
// //     if na(dbg)
// //         dbg := label.new(bar_index, wt1, dbgTxt, style=label.style_label_left, textcolor=color.white, color=color.new(color.black, 50))
// //     else
// //         label.set_xy(dbg, bar_index, wt1)
// //         label.set_text(dbg, dbgTxt)
// // 
// // // =======================================================================================
// // // KASIA — 온차트 퍼포먼스 대시보드 v1.1 (통계 리셋 로직 포함)
// // // =======================================================================================
// // groupDashboard = "UI 대시보드"
// // show_dashboard = input.bool(true, "퍼포먼스 대시보드 표시", group=groupDashboard)
// // 
// // if show_dashboard and barstate.islast
// //     // --- 누적 통계 변수 ---
// //     var int weekly_wins = 0, weekly_trades = 0
// //     var float weekly_pnl = 0.0
// //     var int monthly_wins = 0, monthly_trades = 0
// //     var float monthly_pnl = 0.0
// // 
// //     // --- 마지막으로 계산된 주/월
// //     var int last_calc_week = na
// //     var int last_calc_month = na
// // 
// //     // --- 현재 주/월 (YYYY*100 + week/month)
// //     t_week  = year(time) * 100 + weekofyear(time)
// //     t_month = year(time) * 100 + month(time)
// // 
// //     // 주 교체 시 리셋
// //     if nz(last_calc_week) != t_week
// //         weekly_wins   := 0
// //         weekly_trades := 0
// //         weekly_pnl    := 0.0
// //         last_calc_week := t_week
// // 
// //     // 월 교체 시 리셋
// //     if nz(last_calc_month) != t_month
// //         monthly_wins   := 0
// //         monthly_trades := 0
// //         monthly_pnl    := 0.0
// //         last_calc_month := t_month
// // 
// //     // 신규 거래만 처리
// //     var int last_processed_trade_count = 0
// //     if strategy.closedtrades > last_processed_trade_count
// //         for i = last_processed_trade_count to strategy.closedtrades - 1
// //             trade_pnl = strategy.closedtrades.profit(i)
// //             trade_is_win = trade_pnl > 0
// //             trade_exit_time = strategy.closedtrades.exit_time(i)
// // 
// //             trade_week  = year(trade_exit_time) * 100 + weekofyear(trade_exit_time)
// //             trade_month = year(trade_exit_time) * 100 + month(trade_exit_time)
// // 
// //             if trade_week == t_week
// //                 weekly_trades += 1
// //                 weekly_pnl += trade_pnl
// //                 if trade_is_win
// //                     weekly_wins += 1
// // 
// //             if trade_month == t_month
// //                 monthly_trades += 1
// //                 monthly_pnl += trade_pnl
// //                 if trade_is_win
// //                     monthly_wins += 1
// // 
// //         last_processed_trade_count := strategy.closedtrades
// // 
// //     // --- 파생지표 ---
// //     weekly_win_rate  = weekly_trades  > 0 ? (weekly_wins  / weekly_trades)  * 100.0 : 0.0
// //     monthly_win_rate = monthly_trades > 0 ? (monthly_wins / monthly_trades) * 100.0 : 0.0
// // 
// //     // --- 테이블 생성/업데이트 (우측 상단) ---
// //     var table ui_table = na
// //     if na(ui_table)
// //         ui_table := table.new(position.top_right, 3, 5, border_width=1)
// //     // 헤더
// //     f_fill_cell(ui_table, 0, 0, "KASIA Stats", color.new(color.black, 0))
// //     f_fill_cell(ui_table, 1, 0, "This Week", color.new(color.black, 0))
// //     f_fill_cell(ui_table, 2, 0, "This Month", color.new(color.black, 0))
// // 
// //     // 수익 (PNL)
// //     f_fill_cell(ui_table, 0, 1, "Net Profit", color.new(color.gray, 50))
// //     f_fill_cell(ui_table, 1, 1, "$" + str.tostring(weekly_pnl, format.mintick),  weekly_pnl  > 0 ? color.new(color.teal, 30)   : color.new(color.maroon, 30))
// //     f_fill_cell(ui_table, 2, 1, "$" + str.tostring(monthly_pnl, format.mintick), monthly_pnl > 0 ? color.new(color.teal, 30)   : color.new(color.maroon, 30))
// // 
// //     // 승률 (Win Rate)
// //     f_fill_cell(ui_table, 0, 2, "Win Rate", color.new(color.gray, 50))
// //     f_fill_cell(ui_table, 1, 2, str.tostring(weekly_win_rate, "#.0")  + "%", color.new(color.blue, 50))
// //     f_fill_cell(ui_table, 2, 2, str.tostring(monthly_win_rate, "#.0") + "%", color.new(color.blue, 50))
// // 
// //     // 거래 횟수 (Trades)
// //     f_fill_cell(ui_table, 0, 3, "Total Trades", color.new(color.gray, 50))
// //     f_fill_cell(ui_table, 1, 3, str.tostring(weekly_trades),  color.new(color.gray, 50))
// //     f_fill_cell(ui_table, 2, 3, str.tostring(monthly_trades), color.new(color.gray, 50))
// // 
// // // =======================================================================================
// // // KASIA — 고급 통계 대시보드 v1.0 (좌측 상단)
// // // =======================================================================================
// // show_adv_dashboard = input.bool(true, "고급 통계 대시보드 표시", group=groupDashboard)
// // 
// // if show_adv_dashboard and barstate.islast
// //     float total_gross_profit = strategy.grossprofit
// //     float total_gross_loss   = strategy.grossloss
// //     int   total_trades       = strategy.closedtrades
// //     float max_dd             = strategy.max_drawdown
// // 
// //     // Profit Factor
// //     float profit_factor = total_gross_loss != 0 ? math.abs(total_gross_profit / total_gross_loss) : na
// // 
// //     // Sortino 근사 (손실 거래 기반 하방편차) — 참고용
// //     var float downside_deviation = 0.0
// //     var float avg_return = 0.0
// //     var float sortino_ratio = na
// // 
// //     if total_trades > 1
// //         avg_return := strategy.netprofit / total_trades
// //         var float[] loss_series = array.new_float(0)
// //         for i = 0 to total_trades - 1
// //             float pnl = strategy.closedtrades.profit(i)
// //             if pnl < 0
// //                 array.push(loss_series, pnl)
// // 
// //         if array.size(loss_series) > 1
// //             float sum_sq = 0.0
// //             for v in loss_series
// //                 sum_sq += v * v
// //             downside_deviation := math.sqrt(sum_sq / array.size(loss_series))
// //             sortino_ratio := downside_deviation != 0 ? avg_return / downside_deviation : na
// // 
// //     // 테이블
// //     var table adv_table = na
// //     if na(adv_table)
// //         adv_table := table.new(position.top_left, 2, 5, border_width=1)
// //     // 헤더
// //     f_fill_adv_cell(adv_table, 0, 0, "Advanced Stats", color.new(color.black, 0), true)
// //     table.merge_cells(adv_table, 0, 0, 1, 0)
// // 
// //     // Profit Factor
// //     f_fill_adv_cell(adv_table, 0, 1, "Profit Factor", color.new(color.gray, 50))
// //     f_fill_adv_cell(adv_table, 1, 1, str.tostring(profit_factor, "#.00"), profit_factor > 1.5 ? color.new(color.teal, 30) : color.new(color.orange, 30))
// // 
// //     // Sortino (approx.)
// //     f_fill_adv_cell(adv_table, 0, 2, "Sortino (approx.)", color.new(color.gray, 50))
// //     f_fill_adv_cell(adv_table, 1, 2, str.tostring(sortino_ratio, "#.00"), sortino_ratio > 1 ? color.new(color.teal, 30) : color.new(color.orange, 30))
// // 
// //     // Max Drawdown
// //     f_fill_adv_cell(adv_table, 0, 3, "Max Drawdown", color.new(color.gray, 50))
// //     f_fill_adv_cell(adv_table, 1, 3, "$" + str.tostring(max_dd, format.mintick), color.new(color.maroon, 30))
// // 
// //     // Total Trades
// //     f_fill_adv_cell(adv_table, 0, 4, "Total Trades", color.new(color.gray, 50))
// //     f_fill_adv_cell(adv_table, 1, 4, str.tostring(total_trades), color.new(color.gray, 50))
// // 
// // // =======================================================================================
// // // Metrics / Monitoring — Sortino v1.1 (W/M only, downside deviation fix, perf-limited)
// // // =======================================================================================
// // groupMetrics = "모니터링 (Metrics)"
// // show_metrics       = input.bool(true, "소티노/지표 표시", group=groupMetrics)
// // use_log_return_eq  = input.bool(true,  "로그수익 사용 (Equity)", group=groupMetrics)
// // rf_annual          = input.float(0.0,  "무위험 이자율(연)", group=groupMetrics, step=0.1)
// // max_lookback       = input.int(5000, "최대 Lookback (성능)", group=groupMetrics, tooltip="성능 유지를 위해 지표 계산 시 최대 봉 수를 제한합니다.")
// // 
// // // 올바른 하방편차 계산 (목표수익=rf_per_bar)
// // f_downside_deviation(series, length, target_return) =>
// //     float sum_sq_diff = 0.0
// //     int count = 0
// //     for i = 0 to length - 1
// //         float excess_return = nz(series[i]) - target_return
// //         if excess_return < 0
// //             sum_sq_diff += excess_return * excess_return
// //             count += 1
// //     count > 0 ? math.sqrt(sum_sq_diff / count) : 0.0
// // 
// // _sec_bar      = timeframe.in_seconds(timeframe.period)
// // _bars_day     = _sec_bar > 0 ? math.round(86400.0 / _sec_bar) : 0
// // _bars_week    = math.min(_bars_day * 7,  max_lookback)
// // _bars_month   = math.min(_bars_day * 30, max_lookback)
// // _bars_year    = _sec_bar > 0 ? math.round(31536000.0 / _sec_bar) : 0  // rf 계산용
// // 
// // // per-bar equity returns
// // _eq_ret_raw = use_log_return_eq ? math.log(strategy.equity / nz(strategy.equity[1], strategy.equity))                                 : (strategy.equity / nz(strategy.equity[1], strategy.equity) - 1.0)
// // // per-bar risk-free
// // rf_per_bar = _bars_year > 0 ? (rf_annual/100.0) / _bars_year : 0.0
// // 
// // // 평균 초과수익률 (per-bar)
// // avg_w  = ta.sma(_eq_ret_raw, _bars_week)  - rf_per_bar
// // avg_m  = ta.sma(_eq_ret_raw, _bars_month) - rf_per_bar
// // 
// // // 하방편차 (per-bar)
// // dvol_w = f_downside_deviation(_eq_ret_raw, _bars_week,  rf_per_bar)
// // dvol_m = f_downside_deviation(_eq_ret_raw, _bars_month, rf_per_bar)
// // 
// // // 소티노 (per-bar 기준, 비연율)
// // sortino_w = dvol_w > 0 ? (avg_w / dvol_w) : na
// // sortino_m = dvol_m > 0 ? (avg_m / dvol_m) : na
// // 
// // if show_metrics and barstate.islast
// //     var table met_table = na
// //     if na(met_table)
// //         met_table := table.new(position.top_center, 3, 2, border_width=1)
// //     table.cell(met_table, 0, 0, "Sortino", bgcolor=color.new(color.black, 0), text_color=color.white, text_size=size.small)
// //     table.cell(met_table, 1, 0, "W", bgcolor=color.new(color.black, 0), text_color=color.white, text_size=size.small)
// //     table.cell(met_table, 2, 0, "M", bgcolor=color.new(color.black, 0), text_color=color.white, text_size=size.small)
// // 
// //     table.cell(met_table, 1, 1, na(sortino_w) ? "NA" : str.tostring(sortino_w, "#.00"), bgcolor=color.new(color.gray, 50), text_color=color.white)
// //     table.cell(met_table, 2, 1, na(sortino_m) ? "NA" : str.tostring(sortino_m, "#.00"), bgcolor=color.new(color.gray, 50), text_color=color.white)
// // 
// // // 디버그 라벨에 추가 (있다면)
// // if show_dbg
// // 
// //     dbgTxt := dbgTxt + str.format("\nSortino W/M = {0}/{1}", (na(sortino_w) ? "NA" : str.tostring(sortino_w, "#.00")), (na(sortino_m) ? "NA" : str.tostring(sortino_m, "#.00")))
// //     if not na(dbg)
// //         label.set_text(dbg, dbgTxt)


// ============================================================================
// EXITS — v1.5.6e (2025-09-14 KST) — FADE ONLY + OPPOSITE SIGNAL EXIT
//  - Removes "No-Threshold Opposite Cross" exits
//  - Keeps Opposite-Signal Exit (close-on-opposite-entry-signal)
//  - Fade rules (user-defined):
//      SHORT: after entry at L1/L2, take profit when (wt1 < 0 and golden cross) OR crossing up through -10 with golden cross
//      LONG : after entry below negative threshold, take profit when (wt1 >= 0 and dead cross)
// ============================================================================

// ============================================================================
// EXITS — v1.5.6e (2025-09-14 KST) — FADE ONLY + OPPOSITE SIGNAL EXIT
//  - Removes "No-Threshold Opposite Cross" exits
//  - Keeps Opposite-Signal Exit (close-on-opposite-entry-signal)
//  - Fade rules (user-defined):
//      SHORT: after entry at L1/L2, take profit when (wt1 < 0 and golden cross) OR crossing up through -10 with golden cross
//      LONG : after entry below negative threshold, take profit when (wt1 >= 0 and dead cross)
// ============================================================================

var string __kasia_exit_version = "v1.5.6e: FadeOnly + OppositeSignalExit"

fade_mid_level = input.float(0, "페이드: 중간 레벨(−10 교차 시 추가 페이드)", group=groupExit, step=0.1)
keep_opposite_exit = input.bool(true, "반대 시그널 청산 유지", group=groupExit)

bool longSig  = can_long_L1
bool shortSig = can_short_L1

bool inLong  = strategy.position_size > 0
bool inShort = strategy.position_size < 0

bool short_fade_core = inShort and (wt1 < 0) and bullCross
bool short_fade_mid  = inShort and ta.crossover(wt1, fade_mid_level) and bullCross

if short_fade_core or short_fade_mid
    strategy.close("WT_Short", comment=short_fade_mid ? "WT Fade Exit (−10 GoldenCross)" : "WT Fade Exit (NegZone GoldenCross)")

bool long_fade = inLong and (wt1 >= 0) and bearCross
if long_fade
    strategy.close("WT_Long", comment="WT Fade Exit (>=0 DeadCross)")

if keep_opposite_exit
    if inLong and shortSig
        strategy.close("WT_Long", comment="Opposite Signal Exit → Short")
        if allow_reverse
            strategy.entry("WT_Short", strategy.short, qty=qty_eff)
            
    if inShort and longSig
        strategy.close("WT_Short", comment="Opposite Signal Exit → Long")
        if allow_reverse
            strategy.entry("WT_Long", strategy.long, qty=qty_eff)

 5.5번스크립트 스캘핑 풀백

//@version=5
// ============================================================================
//  Scalping PullBack Tool R1.1 — STRATEGY (Pine v5)  [KASIA FIX4]
//  Base: "Scalping PullBack Tool R1.1 by JustUncleL" (study → strategy conversion)
//  This FIX4 focuses on *entries not firing* on some charts. Changes keep the
//  original behavior but harden session/time, HA data requests, and entry gating.
//  Full version — no feature removal.
// ============================================================================
//  CHANGE LOG — 2025-09-09 KST (KASIA v3-fix4)
//  • Session: default to \"24x7\" and simplify session check via time(..., session).
//  • HA data: request.security(ticker.heikinashi, timeframe.period, ..., gaps=off, lookahead=off).
//  • Entry gating: remove over‑restrictive canSizeNow gate; rely on pyramiding=0.
//  • PAC cross lookback: add toggle to disable the barssince() gate (more permissive).
//  • Debug: plotshape(Long/Short raw) to visually verify conditions.
//  • Minor: defensive NA guards on priceSrc & qty calc to prevent zero/NA qty.
// ============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// STRATEGY HEADER
// ─────────────────────────────────────────────────────────────────────────────
strategy(title="Scalping PullBack Tool R1.1 — Strategy (KASIA v5 qty+lev FIX4)",
     shorttitle="SCALPTOOL R1.1 — STRAT (KASIA v5 FIX4)",
     overlay=true,
     initial_capital=10000,
     commission_type=strategy.commission.percent,
     commission_value=0.06,                 // 커미션 0.06%
     slippage=1,
     pyramiding=0,
     calc_on_order_fills=true,
     calc_on_every_tick=false,
     process_orders_on_close=true,
     default_qty_type=strategy.fixed,       // qty 직접 지정
     default_qty_value=0,
     margin_long=100,                       // 1x (레버리지는 qty에 반영)
     margin_short=100)

// ============================================================================
//  INPUTS — original + presets + sizing + time
// ============================================================================
group_base   = "Base Settings"
HiLoLen_inp         = input.int(34,  minval=2, title="(Manual) High-Low PAC Length", group=group_base)
fastEMAlength_inp   = input.int(89,  minval=2, title="(Manual) Fast EMA length",    group=group_base)
mediumEMAlength_inp = input.int(200, minval=2, title="(Manual) Medium EMA length",  group=group_base)
slowEMAlength_inp   = input.int(600, minval=2, title="(Manual) Slow EMA length",    group=group_base)
ShowFastEMA     = input.bool(true,  title="Show Fast EMA",   group=group_base)
ShowMediumEMA   = input.bool(true,  title="Show Medium EMA", group=group_base)
ShowSlowEMA     = input.bool(false, title="Show Slow EMA",   group=group_base)
ShowHHLL        = input.bool(false, title="Show HH/LL labels", group=group_base)
ShowFractals    = input.bool(true,  title="Show Fractals",     group=group_base)
filterBW        = input.bool(false, title="Use Regular(ON)/BW(OFF) Fractal Filter", group=group_base)
ShowBarColor    = input.bool(true,  title="Show coloured Bars around PAC", group=group_base)
ShowBuySell     = input.bool(true,  title="Show Buy/Sell Alert Arrows",     group=group_base)
Lookback        = input.int(3,      minval=1, title="Pullback Lookback for PAC Cross Check", group=group_base)
DelayArrow      = input.bool(true,  title="Use Closed-Candle (confirmed) entries (권장)",     group=group_base)
ShowTrendBGcolor= input.bool(true,  title="Show Trend Background Color", group=group_base)
UseHAcandles    = input.bool(true,  title="Use Heikin Ashi candles in Algo Calculations", group=group_base)

// Presets (two only)
group_preset = "Presets (Lengths)"
usePresets   = input.bool(true, title="Use Presets (1m/5m)", group=group_preset)
presetTF     = input.string("1m", options=["1m", "5m"], title="Preset timeframe", group=group_preset)

// Effective lengths (either preset or manual)
int effHiLoLen   = usePresets ? (presetTF == "1m" ? 21  : 34 )  : HiLoLen_inp
int effFastLen   = usePresets ? (presetTF == "1m" ? 55  : 89 )  : fastEMAlength_inp
int effMedLen    = usePresets ? (presetTF == "1m" ? 144 : 200)  : mediumEMAlength_inp
int effSlowLen   = usePresets ? (presetTF == "1m" ? 377 : 600)  : slowEMAlength_inp

// Strategy-only Risk/Exit Inputs
groupRisk   = "Strategy — Risk / Exits"
useTP       = input.bool(true,  title="Use Take Profit (%)",   group=groupRisk)
tpPerc      = input.float(0.35, title="TP (%)", step=0.01,     group=groupRisk)  // 예: 0.35%
useSL       = input.bool(true,  title="Use Stop Loss (%)",     group=groupRisk)
slPerc      = input.float(0.25, title="SL (%)", step=0.01,     group=groupRisk)  // 예: 0.25%
usePACExit  = input.bool(true,  title="Use PAC Centre Re-Cross Exit", group=groupRisk)

oppGroup    = "Strategy — Opposite Signal Handling"
useOppExit  = input.bool(true,  title="Close on Opposite Signal",      group=oppGroup)
allowRev    = input.bool(false, title="Allow Reversal (Switch Position)", group=oppGroup)

// ── Qty & Leverage sizing
groupQty  = "Strategy — Quantity / Leverage"
useFixedQty  = input.bool(false,        title="Use Fixed $ Size (OFF → Equity % mode)", group=groupQty)
fixedCash    = input.float(3000.0,      title="Fixed $ Size", step=1.0,                 group=groupQty)
qtyPct       = input.float(30.0,        title="Equity % Size (복리)", step=0.1,         group=groupQty)
leverage     = input.float(10.0,        title="Leverage (sizing only)", step=0.1,       group=groupQty)
priceSrc     = input.source(close,      title="Price for Qty calc",                      group=groupQty)

// ── Time / Session (time filter only)
groupTime = "Backtest / Time Filter"
btStart   = input.time(defval=timestamp("01 Jan 2024 00:00 +0000"), title="Backtest Start (exchange tz)", group=groupTime)
useSession= input.bool(false,            title="Use intraday Session filter (기본 OFF: 24x7)", group=groupTime)
sessionStr= input.session("24x7",        title="Session (e.g., 0900-1700:1234567 or 24x7)",  group=groupTime)
inSession = useSession ? not na(time(timeframe.period, sessionStr)) : true
inBacktest= time >= btStart
timeOK    = inSession and inBacktest

// Entry strictness
groupEntry = "Entry Strictness"
useLookbackGate = input.bool(true, title="Require recent pullback (barssince gate)", group=groupEntry)

// Debug
group_dbg   = "Debug"
ShowDebug   = input.bool(false, title="Show debug plots", group=group_dbg)

// Derived
int Delay = DelayArrow ? 1 : 0

// ============================================================================
//  BASE FUNCTIONS — v5 (using ta.*)
// ============================================================================
var string haTicker = ticker.heikinashi(syminfo.tickerid)
haClose = UseHAcandles ? request.security(haTicker, timeframe.period, close,  gaps=barmerge.gaps_off, lookahead=barmerge.lookahead_off) : close
haOpen  = UseHAcandles ? request.security(haTicker, timeframe.period, open,   gaps=barmerge.gaps_off, lookahead=barmerge.lookahead_off) : open
haHigh  = UseHAcandles ? request.security(haTicker, timeframe.period, high,   gaps=barmerge.gaps_off, lookahead=barmerge.lookahead_off) : high
haLow   = UseHAcandles ? request.security(haTicker, timeframe.period, low,    gaps=barmerge.gaps_off, lookahead=barmerge.lookahead_off) : low

isRegularFractal(mode) =>
    ret = mode == 1 ? high[4] < high[3] and high[3] < high[2] and high[2] > high[1] and high[1] > high[0] :
          mode == -1 ? low[4]  > low[3]  and low[3]  > low[2]  and low[2]  < low[1]  and low[1]  < low[0] :
          false
    ret

isBWFractal(mode) =>
    ret = mode == 1 ? high[4] < high[2] and high[3] <= high[2] and high[2] >= high[1] and high[2] >  high[0] :
          mode == -1 ? low[4]  > low[2]  and low[3]  >= low[2]  and low[2]  <= low[1]  and low[2]  <  low[0] :
          false
    ret

// ============================================================================
//  SERIES SETUP
// ============================================================================
fastEMA   = ta.ema(haClose, effFastLen)
mediumEMA = ta.ema(haClose, effMedLen)
slowEMA   = ta.ema(haClose, effSlowLen)
pacC      = ta.ema(haClose, effHiLoLen)
pacL      = ta.ema(haLow,   effHiLoLen)
pacU      = ta.ema(haHigh,  effHiLoLen)

TrendDirection = fastEMA > mediumEMA and pacL > mediumEMA ? 1 :
                 fastEMA < mediumEMA and pacU < mediumEMA ? -1 : 0

filteredtopf = filterBW ? isRegularFractal(1)   : isBWFractal(1)
filteredbotf = filterBW ? isRegularFractal(-1)  : isBWFractal(-1)

valuewhen_H0 = ta.valuewhen(filteredtopf == true, high[2], 0)
valuewhen_H1 = ta.valuewhen(filteredtopf == true, high[2], 1)
valuewhen_H2 = ta.valuewhen(filteredtopf == true, high[2], 2)

higherhigh = filteredtopf == false ? false : valuewhen_H1 < valuewhen_H0 and valuewhen_H2 < valuewhen_H0
lowerhigh  = filteredtopf == false ? false : valuewhen_H1 > valuewhen_H0 and valuewhen_H2 > valuewhen_H0

valuewhen_L0 = ta.valuewhen(filteredbotf == true, low[2], 0)
valuewhen_L1 = ta.valuewhen(filteredbotf == true, low[2], 1)
valuewhen_L2 = ta.valuewhen(filteredbotf == true, low[2], 2)

higherlow = filteredbotf == false ? false : valuewhen_L1 < valuewhen_L0 and valuewhen_L2 < valuewhen_L0
lowerlow  = filteredbotf == false ? false : valuewhen_L1 > valuewhen_L0 and valuewhen_L2 > valuewhen_L0

// ============================================================================
//  PLOTTING (retain visuals; transp → color.new)
// ============================================================================
L = plot(pacL, color=color.new(color.gray, 50), linewidth=1, title="High PAC EMA")
U = plot(pacU, color=color.new(color.gray, 50), linewidth=1, title="Low PAC EMA")
C = plot(pacC, color=color.new(color.red,   0), linewidth=2, title="Close PAC EMA")
fill(L, U, color=color.new(color.gray, 90), title="Fill HiLo PAC")

BARcolor = haClose > pacU ? color.new(color.blue, 0) : haClose < pacL ? color.new(color.red, 0) : color.new(color.gray, 0)
barcolor(ShowBarColor ? BARcolor : na, title="Bar Colours")

BGcolor = TrendDirection == 1 ? color.new(color.green, 80) : TrendDirection == -1 ? color.new(color.red, 80) : color.new(color.yellow, 80)
bgcolor(ShowTrendBGcolor ? BGcolor : na, title="Trend BG Color")

plot(ShowFastEMA   ? fastEMA   : na, color=color.new(color.green, 20), linewidth=2, title="fastEMA")
plot(ShowMediumEMA ? mediumEMA : na, color=color.new(color.blue,  20), linewidth=3, title="mediumEMA")
plot(ShowSlowEMA   ? slowEMA   : na, color=color.new(color.black, 20), linewidth=4, title="slowEMA")

plotshape(ShowFractals ? filteredtopf : na, title='Filtered Top Fractals', style=shape.triangledown, location=location.abovebar, color=color.new(color.red, 0),  offset=-2)
plotshape(ShowFractals ? filteredbotf : na, title='Filtered Bottom Fractals', style=shape.triangleup,   location=location.belowbar, color=color.new(color.lime, 0), offset=-2)

plotshape(ShowHHLL ? higherhigh : na, title='Higher High', style=shape.square,   location=location.abovebar, color=color.new(color.maroon, 0), text="[HH]", offset=-2)
plotshape(ShowHHLL ? lowerhigh  : na, title='Lower High',  style=shape.square,   location=location.abovebar, color=color.new(color.maroon, 0), text="[LH]", offset=-2)
plotshape(ShowHHLL ? higherlow  : na, title='High Low',    style=shape.square,   location=location.belowbar, color=color.new(color.green, 0),  text="[HL]", offset=-2)
plotshape(ShowHHLL ? lowerlow   : na, title='Lower Low',   style=shape.square,   location=location.belowbar, color=color.new(color.green, 0),  text="[LL]", offset=-2)

plot(ShowDebug ? ta.barssince(haClose < pacC) : na, color=color.new(color.gray, 100), title="bars since haClose<pacC>")
plot(ShowDebug ? ta.barssince(haClose > pacC) : na, color=color.new(color.gray, 100), title="bars since haClose>pacC>")

// ============================================================================
//  SIGNAL STATE & ALERTS (original logic, v5-safe) + permissive toggle
// ============================================================================
var int TradeDirection = 0
TradeDirection := nz(TradeDirection[1], 0)

pacExitU_strict = haOpen < pacU and haClose > pacU and ta.barssince(haClose < pacC) <= Lookback
pacExitL_strict = haOpen > pacL and haClose < pacL and ta.barssince(haClose > pacC) <= Lookback
pacExitU_loose  = haOpen < pacU and haClose > pacU
pacExitL_loose  = haOpen > pacL and haClose < pacL

pacExitU = useLookbackGate ? pacExitU_strict : pacExitU_loose
pacExitL = useLookbackGate ? pacExitL_strict : pacExitL_loose

Buy  = TrendDirection == 1  and pacExitU
Sell = TrendDirection == -1 and pacExitL

TradeDirection := TradeDirection == 1  and haClose < pacC ? 0 :
                  TradeDirection == -1 and haClose > pacC ? 0 :
                  TradeDirection == 0 and Buy  ? 1 :
                  TradeDirection == 0 and Sell ? -1 : TradeDirection

// ── Arrow series (no variable history index; single-line ternary)
arrowPrevZero = DelayArrow ? nz(TradeDirection[2]) == 0 : nz(TradeDirection[1]) == 0
arrowNowDir   = DelayArrow ? TradeDirection[1]          : TradeDirection
arrowSeries   = ShowBuySell and arrowPrevZero and (arrowNowDir != 0) ? arrowNowDir : na
plotarrow(arrowSeries,
          offset=-(DelayArrow ? 1 : 0),
          colorup=color.new(color.green, 0), colordown=color.new(color.maroon, 0),
          minheight=20, maxheight=50, title="Buy/Sell Arrow")

Long  = nz(TradeDirection[1]) == 0 and TradeDirection == 1
Short = nz(TradeDirection[1]) == 0 and TradeDirection == -1

// Visual debug of entry conditions
plotshape(ShowDebug and Long,  title="Long RAW",  style=shape.circle, location=location.belowbar, color=color.new(color.green, 0), size=size.tiny, text="L")
plotshape(ShowDebug and Short, title="Short RAW", style=shape.circle, location=location.abovebar, color=color.new(color.red,   0), size=size.tiny, text="S")

alertcondition(Long,  title="Buy Condition",  message="BUY")
alertcondition(Short, title="Sell Condition", message="SELL")

// ============================================================================
//  QTY SIZING (fixed $ or equity %) + leverage; compounding via equity%
// ============================================================================
srcPrice = na(priceSrc) ? close : priceSrc
capBase  = useFixedQty ? fixedCash : (strategy.equity * (qtyPct / 100.0))
desiredNotional = capBase * leverage
calc_qty = desiredNotional / math.max(srcPrice, syminfo.mintick)

// ============================================================================
//  STRATEGY ENTRIES & EXITS
// ============================================================================
enterLongRaw  = Long  and barstate.isconfirmed
enterShortRaw = Short and barstate.isconfirmed

// Rely on pyramiding=0 to prevent stacking. We do NOT gate by opentrades here,
// so signals on the first available bar can actually submit.
if timeOK
    if enterLongRaw
        if allowRev
            strategy.entry("L", strategy.long, qty=calc_qty, comment="PB-Recovery LONG (rev)")
        else
            if useOppExit and strategy.position_size < 0
                strategy.close("S", comment="Opposite Exit")
            if strategy.position_size == 0
                strategy.entry("L", strategy.long, qty=calc_qty, comment="PB-Recovery LONG")

    if enterShortRaw
        if allowRev
            strategy.entry("S", strategy.short, qty=calc_qty, comment="PB-Recovery SHORT (rev)")
        else
            if useOppExit and strategy.position_size > 0
                strategy.close("L", comment="Opposite Exit")
            if strategy.position_size == 0
                strategy.entry("S", strategy.short, qty=calc_qty, comment="PB-Recovery SHORT")

// %TP/%SL
longTP  = useTP ? strategy.position_avg_price * (1 + tpPerc/100.0) : na
longSL  = useSL ? strategy.position_avg_price * (1 - slPerc/100.0) : na
shortTP = useTP ? strategy.position_avg_price * (1 - tpPerc/100.0) : na
shortSL = useSL ? strategy.position_avg_price * (1 + slPerc/100.0) : na

if strategy.position_size > 0
    strategy.exit(id="XL", from_entry="L", limit=longTP, stop=longSL, comment="TP/SL Long")
if strategy.position_size < 0
    strategy.exit(id="XS", from_entry="S", limit=shortTP, stop=shortSL, comment="TP/SL Short")

// PAC centre re-cross exit
exitLongPAC  = usePACExit and strategy.position_size > 0 and haClose < pacC and barstate.isconfirmed
exitShortPAC = usePACExit and strategy.position_size < 0 and haClose > pacC and barstate.isconfirmed
if exitLongPAC
    strategy.close("L", comment="PAC Re-Cross")
if exitShortPAC
    strategy.close("S", comment="PAC Re-Cross")

// Opposite signal exit (no reversal)
if useOppExit and not allowRev
    if strategy.position_size > 0 and enterShortRaw and timeOK
        strategy.close("L", comment="Opposite Signal")
    if strategy.position_size < 0 and enterLongRaw and timeOK
        strategy.close("S", comment="Opposite Signal")

// ============================================================================
//  END
// ============================================================================

6. 6번스크립트 MAX PRO R3
//@version=5
strategy(
     title = "KASIA Regime",
     overlay = true,
     max_bars_back = 5000,

     // === 자본/수수료/슬리피지 ===
     initial_capital  = input.float(10000, "초기자본(USD)", minval = 1, group = "A. 기본/운영"),
     commission_type  = strategy.commission.percent,
     commission_value = input.bool(true, "수수료 0.05% 적용?", group = "A. 기본/운영") ? 0.05 : 0.0,
     slippage         = input.int(0, "슬리피지(틱)", minval = 0, group = "A. 기본/운영"),

     // === 기본 주문 크기(%) ===
     default_qty_type  = strategy.percent_of_equity,
     default_qty_value = input.float(100, "기본 주문 크기(% of equity)", minval = 0.1, maxval = 10000, group = "A. 기본/운영"),

     // === 레버리지(마진%) ===  // margin_% = 100 / leverage_x
     margin_long  = input.bool(true, "레버리지(롱) 사용?", group = "A. 기본/운영")
                     ? (100.0 / input.float(5.0, "롱 레버리지(x)",  minval = 1.0, group = "A. 기본/운영"))
                     : 100.0,
     margin_short = input.bool(true, "레버리지(숏) 사용?", group = "A. 기본/운영")
                     ? (100.0 / input.float(5.0, "숏 레버리지(x)", minval = 1.0, group = "A. 기본/운영"))
                     : 100.0,

     pyramiding = 4,
     calc_on_every_tick = false,
     process_orders_on_close = true
     )   

// =========================[ A. 심볼/기본/날짜 ]========================
symBase      = input.symbol("BINANCE:SOLUSDT.P", "기본 심볼", group="A. 기본/운영")
maxBarsHold  = input.int(0, "최대 보유 바 수(0=무제한)", minval=0, group="A. 기본/운영")
// 백테스트 시작일 (시작일만)
useStart = input.bool(true, "백테스트 시작일 사용?", group="A. 기본/운영")
startAt  = input.time(timestamp("2024-01-01 00:00 +0000"), "시작일(UTC)", group="A. 기본/운영")
inDate   = not useStart or (time >= startAt)

// 경고 라벨
isSOLP = str.contains(syminfo.ticker, "SOLUSDT.P") and str.contains(syminfo.ticker, "BINANCE")
if not isSOLP
    label.new(bar_index, high, "경고: 기본은 BINANCE:SOLUSDT.P.\n다른 심볼이면 파라미터 재점검 권장.", style=label.style_label_down, textcolor=color.white, color=color.new(color.red, 0))

// ======================[ B. UTC/상위봉/세션 ]=====================
useUTCgate   = input.bool(false, "UTC 진입 제한 사용", group="B. UTC/상위봉/세션")
utcStartH    = input.int(7,  "UTC 시작-시(0~23)", minval=0, maxval=23, group="B. UTC/상위봉/세션")
utcStartM    = input.int(0,  "UTC 시작-분(0~59)", minval=0, maxval=59, group="B. UTC/상위봉/세션")
utcEndH      = input.int(21, "UTC 종료-시(0~23)", minval=0, maxval=23, group="B. UTC/상위봉/세션")
utcEndM      = input.int(0,  "UTC 종료-분(0~59)", minval=0, maxval=59, group="B. UTC/상위봉/세션")
utcNowMin = hour(time, "UTC")*60 + minute(time, "UTC")
utcStartMin = utcStartH*60 + utcStartM
utcEndMin   = utcEndH*60 + utcEndM
utcOK = not useUTCgate or (utcStartMin <= utcEndMin ? (utcNowMin >= utcStartMin and utcNowMin <= utcEndMin)
                                                 : (utcNowMin >= utcStartMin or  utcNowMin <= utcEndMin))

useFundingEmbargo = input.bool(false, "펀딩 롤오버 금지창(00/08/16 UTC ±N분)", group="B. UTC/상위봉/세션")
embargoMin        = input.int(5, "금지창 분(±)", minval=0, maxval=30, group="B. UTC/상위봉/세션")
fundHours = array.from(0, 8, 16)
hUTC = hour(time, "UTC"), mUTC = minute(time, "UTC")
isFundHour = array.includes(fundHours, hUTC)
embargoOK = not useFundingEmbargo or not (isFundHour and (mUTC <= embargoMin or mUTC >= 60-embargoMin))

useMTF      = input.bool(true, "상위봉 필터 사용(EMA+RSI)", group="B. UTC/상위봉/세션")
tf_htf      = input.timeframe("60", "상위봉 TF (예: 60/240)", group="B. UTC/상위봉/세션")
sym     = syminfo.tickerid
htfEMA  = request.security(sym, tf_htf, ta.ema(close, 50),  barmerge.gaps_off, barmerge.lookahead_off)
htfRSI  = request.security(sym, tf_htf, ta.rsi(close, 14),   barmerge.gaps_off, barmerge.lookahead_off)
htfOK   = not useMTF or (barstate.isconfirmed and close > htfEMA and htfEMA > nz(htfEMA[1]) and htfRSI >= 50)

useSessions  = input.bool(false, "세션 필터(아시아/런던/뉴욕)", group="B. UTC/상위봉/세션")
asiaSess     = input.session("0000-0900", "아시아(차트TZ)", group="B. UTC/상위봉/세션")
lonSess      = input.session("0700-1600", "런던(차트TZ)", group="B. UTC/상위봉/세션")
nySess       = input.session("1300-2200", "뉴욕(차트TZ)", group="B. UTC/상위봉/세션")
avoidWeekend = input.bool(false, "주말 진입 회피", group="B. UTC/상위봉/세션")
inAsia = not useSessions or not na(time(timeframe.period, asiaSess))
inLon  = not useSessions or not na(time(timeframe.period, lonSess))
inNY   = not useSessions or not na(time(timeframe.period, nySess))
inSess = not useSessions or (inAsia or inLon or inNY)
isWeekend = (dayofweek == dayofweek.saturday or dayofweek == dayofweek.sunday)

// ======================[ C. 오실레이터/AVWAP/Anchors ]=====================
useAVWAP     = input.bool(false, "세션 VWAP 필터", group="C. 오실/AVWAP/앵커")
rsiLen       = input.int(7, "RSI 기간(레인지)", group="C. 오실/AVWAP/앵커")
stochLen     = input.int(14, "스토캐스틱 K 기간", group="C. 오실/AVWAP/앵커")
stochSig     = input.int(3,  "스토캐스틱 D 기간", group="C. 오실/AVWAP/앵커")
sessVWAP = ta.vwap
vwapOK_Long  = not useAVWAP or close >= sessVWAP
vwapOK_Short = not useAVWAP or close <= sessVWAP

// Anchored VWAP (일/주/월/커스텀 앵커)
useAnchors = input.bool(false, "Anchored VWAP(AVWAP) 사용", group="C. 오실/AVWAP/앵커")
anchorDaily = input.bool(true, "일봉 시작 앵커", group="C. 오실/AVWAP/앵커")
anchorWeekly= input.bool(false,"주봉 시작 앵커", group="C. 오실/AVWAP/앵커")
anchorMonthly=input.bool(false,"월봉 시작 앵커", group="C. 오실/AVWAP/앵커")
useCustomAnchor = input.bool(false, "커스텀 앵커 사용", group="C. 오실/AVWAP/앵커")
customAnchorAt  = input.time(timestamp("2024-01-01 00:00 +0000"), "커스텀 앵커(UTC)", group="C. 오실/AVWAP/앵커")
var float num = na
var float den = na
newDay  = ta.change(time("D"))
newWeek = ta.change(time("W"))
newMonth= ta.change(time("M"))
resetAVWAP = (useAnchors and ((anchorDaily and newDay) or (anchorWeekly and newWeek) or (anchorMonthly and newMonth))) or (useCustomAnchor and time == customAnchorAt)
if na(num) or resetAVWAP
    num := hlc3*volume
    den := volume
else
    num += hlc3*volume
    den += volume
anchoredVWAP = useAnchors ? (den>0? num/den : na) : na

// ======================[ H. WFA 파라미터 뱅크 ]=====================
useWFA    = input.bool(true, "WFA 파라미터 뱅크", group="H. WFA")
wfaDays   = input.int(30, "세트 기간(일)", minval=5, group="H. WFA")
wfaSets   = input.int(3, "세트 개수", minval=1, maxval=5, group="H. WFA")
var float[] A_trend = array.from(26.0, 24.0, 28.0)
var float[] A_range = array.from(18.0, 20.0, 16.0)
var float[] A_bbMul = array.from(2.0,  2.2,  1.8)
var float[] A_qMult = array.from(0.90, 0.85, 0.95)
var float[] A_vMult = array.from(1.30, 1.60, 1.40)
var float[] A_atrTm = array.from(2.2,  1.8,  2.8)
var float[] A_emaF  = array.from(20.0, 13.0, 34.0)
var float[] A_emaS  = array.from(50.0, 55.0, 89.0)
var float[] A_rsiL  = array.from(25.0, 30.0, 28.0)
var float[] A_rsiH  = array.from(75.0, 70.0, 72.0)
var float[] A_don   = array.from(20.0, 30.0, 14.0)
wfaIdx() =>
    if not useWFA
        0
    else
        winMS = int(wfaDays) * 24 * 60 * 60 * 1000
        seg   = int(math.floor((time - startAt) / winMS))
        math.max(0, seg % wfaSets)
idx = wfaIdx()

// ======================[ D. 레짐/트렌드 + 히스테리시스 ]=====================
adxLen      = input.int(14, "ADX 기간", group="D. 레짐/트렌드/히스테리시스")
adxTrend_i  = input.float(25.0, "추세장: ADX ≥", group="D. 레짐/트렌드/히스테리시스")
adxRange_i  = input.float(18.0, "레인지: ADX ≤", group="D. 레짐/트렌드/히스테리시스")
emaFastLen_i= input.int(20, "EMA Fast", group="D. 레짐/트렌드/히스테리시스")
emaSlowLen_i= input.int(50, "EMA Slow", group="D. 레짐/트렌드/히스테리시스")
trendFilter  = input.string("EMA", "주 추세필터", options=["EMA","Supertrend","KAMA"], group="D. 레짐/트렌드/히스테리시스")
stFactor     = input.float(2.0, "Supertrend ATR factor", group="D. 레짐/트렌드/히스테리시스")
stPeriod     = input.int(10,   "Supertrend ATR period", group="D. 레짐/트렌드/히스테리시스")
kamaLen      = input.int(10,   "KAMA length", group="D. 레짐/트렌드/히스테리시스")

// Effective params from WFA
adxTrend     = useWFA ? array.get(A_trend, idx) : adxTrend_i
adxRange     = useWFA ? array.get(A_range, idx) : adxRange_i
bbLenEff     = input.int(20, "BB 길이", group="E. 스퀴즈/NR/돈키안/ORB/캔들")
bbMultEff    = useWFA ? array.get(A_bbMul, idx) : input.float(2.0, "BB 배수", step=0.1, group="E. 스퀴즈/NR/돈키안/ORB/캔들")
quietMult    = useWFA ? array.get(A_qMult, idx) : input.float(0.80, "Quiet if BBW < MA*quietMult", group="E. 스퀴즈/NR/돈키안/ORB/캔들")
volatileMult = useWFA ? array.get(A_vMult, idx) : input.float(1.50, "Volatile if BBW > MA*volatileMult", group="E. 스퀴즈/NR/돈키안/ORB/캔들")
atrTrailMult = useWFA ? array.get(A_atrTm, idx) : 2.0
emaFastLen   = useWFA ? int(array.get(A_emaF, idx)) : emaFastLen_i
emaSlowLen   = useWFA ? int(array.get(A_emaS, idx)) : emaSlowLen_i
rsiLow       = useWFA ? int(array.get(A_rsiL, idx)) : 30
rsiHigh      = useWFA ? int(array.get(A_rsiH, idx)) : 70
donLenEff    = useWFA ? int(array.get(A_don,  idx)) : 20

emaFast   = ta.ema(close, emaFastLen)
emaSlow   = ta.ema(close, emaSlowLen)
atrLen    = input.int(14, "ATR 기간", group="D. 레짐/트렌드/히스테리시스")
atrVal    = ta.atr(atrLen)

float stLine = na
int   stDir  = na
if trendFilter == "Supertrend"
    [stLine, stDir] := ta.supertrend(stFactor, stPeriod)
kama = trendFilter == "KAMA" ? ta.kama(close, kamaLen) : na
trendUp_now   = trendFilter == "EMA"        ? (close > emaSlow and emaFast > emaSlow) :
                trendFilter == "Supertrend" ? (stDir == -1) :
                (close > kama)
trendDown_now = trendFilter == "EMA"        ? (close < emaSlow and emaFast < emaSlow) :
                trendFilter == "Supertrend" ? (stDir == 1) :
                (close < kama)

// 히스테리시스: 상태 전환에 확증 바 요구
confirmBars = input.int(2, "레짐 전환 확증 바 수", minval=0, group="D. 레짐/트렌드/히스테리시스")
var int trendUpScore=0, trendDownScore=0
trendUpScore   := trendUp_now   ? math.min(confirmBars, trendUpScore+1)   : 0
trendDownScore := trendDown_now ? math.min(confirmBars, trendDownScore+1) : 0
trendUp   = trendUpScore   >= confirmBars
trendDown = trendDownScore >= confirmBars

// ======================[ E. 스퀴즈/NR/돈키안/ORB/캔들 ]=====================
groupTrig = "E. 스퀴즈/NR/돈키안/ORB/캔들"
bbwLookback = input.int(50, "BBW 평균 기간", group=groupTrig)
kcLen        = input.int(20,     "Keltner EMA 길이", group=groupTrig)
kcMult       = input.float(1.5,  "Keltner ATR 배수", group=groupTrig)
useTTMSq     = input.bool(true,  "TTM Squeeze 확인(BB in KC)", group=groupTrig)
useQuiet2Vol = input.bool(true,  "Quiet→Vol 전환 요구", group=groupTrig)
useNR   = input.bool(true, "NR7/IDNR4", group=groupTrig)
useORB  = input.bool(false,"오프닝 레인지 브레이크(UTC)", group=groupTrig)
orbMin  = input.int(15, "오프닝 범위 분", minval=1, group=groupTrig)
useCandles = input.bool(true, "캔들(엔걸핑/핀바)", group=groupTrig)
useDonchian  = input.bool(true, "돈키안 돌파(추세장)", group=groupTrig)

bbLen = bbLenEff
bbMid     = ta.sma(close, bbLen)
bbUp      = bbMid + bbMultEff * ta.stdev(close, bbLen)
bbDn      = bbMid - bbMultEff * ta.stdev(close, bbLen)
bbw       = (bbUp - bbDn) / bbMid
bbwMA     = ta.sma(bbw, bbwLookback)
isQuiet   = bbw < bbwMA * quietMult
isVolNow  = bbw > bbwMA * volatileMult
kcMid   = ta.ema(close, kcLen)
kcUp    = kcMid + kcMult * atrVal
kcDn    = kcMid - kcMult * atrVal
ttmSqueezeOn = (bbUp < kcUp and bbDn > kcDn)
rng(b) => high[b] - low[b]
isNR7   = useNR and (rng(0) <= ta.lowest(rng(0), 7))
isInside= useNR and (high <= high[1] and low >= low[1])
isNR4   = useNR and (rng(0) <= ta.lowest(rng(0), 4))
isIDNR4 = useNR and isInside and isNR4
dayStartUTC = timestamp("UTC", year, month, dayofmonth, 0, 0)
isORBWindow = useORB ? (time - dayStartUTC) <= orbMin * 60 * 1000 : true
donLen = donLenEff
donU = ta.highest(high, donLen)
donL = ta.lowest(low,  donLen)
body   = math.abs(close-open)
upperW = high - math.max(close, open)
lowerW = math.min(close, open) - low
bullEng = useCandles and (close>open and close[1]<open[1] and close>=open[1] and open<=close[1])
bearEng = useCandles and (close<open and close[1]>open[1] and close<=open[1] and open>=close[1])
hammer   = useCandles and (lowerW >= body*2 and upperW <= body and close>open)
shooting = useCandles and (upperW >= body*2 and lowerW <= body and close<open)
patternBull = bullEng or hammer
patternBear = bearEng or shooting
q2vUp   = isVolNow and isQuiet[1]  and close > bbUp
q2vDown = isVolNow and isQuiet[1]  and close < bbDn

// ======================[ F. 크로스섹셔널 + RV 게이트 ]=====================
useCS        = input.bool(true, "크로스섹셔널 필터", group="F. 크로스섹셔널/RV")
csTF         = input.timeframe("60", "CS TF", group="F. 크로스섹셔널/RV")
csMomLen     = input.int(24, "모멘텀 Lookback(바)", group="F. 크로스섹셔널/RV")
csRVlen      = input.int(96, "리얼라이즈드 분산 윈도우(바)", group="F. 크로스섹셔널/RV")
csTopN       = input.int(4,  "롱: 모멘텀 상위 N", minval=1, group="F. 크로스섹셔널/RV")
csBottomN    = input.int(4,  "숏: 모멘텀 하위 N", minval=1, group="F. 크로스섹셔널/RV")
useCS_RVlow  = input.bool(true,  "롱: 저분산 선호", group="F. 크로스섹셔널/RV")
rvLowTopN    = input.int(5, "롱: 저분산 상위 N 컷", minval=1, group="F. 크로스섹셔널/RV")
useCS_RVhigh = input.bool(false, "숏: 고분산 선호", group="F. 크로스섹셔널/RV")
rvHighTopN   = input.int(5, "숏: 고분산 상위 N 컷", minval=1, group="F. 크로스섹셔널/RV")
cs1  = input.symbol("BINANCE:BTCUSDT.P", "CS #1", group="F. 크로스섹셔널/RV")
cs2  = input.symbol("BINANCE:ETHUSDT.P", "CS #2", group="F. 크로스섹셔널/RV")
cs3  = input.symbol("BINANCE:SOLUSDT.P", "CS #3(타깃)", group="F. 크로스섹셔널/RV")
cs4  = input.symbol("BINANCE:BNBUSDT.P", "CS #4", group="F. 크로스섹셔널/RV")
cs5  = input.symbol("BINANCE:XRPUSDT.P", "CS #5", group="F. 크로스섹셔널/RV")
cs6  = input.symbol("BINANCE:ADAUSDT.P", "CS #6", group="F. 크로스섹셔널/RV")
cs7  = input.symbol("BINANCE:DOGEUSDT.P","CS #7", group="F. 크로스섹셔널/RV")
cs8  = input.symbol("BINANCE:LINKUSDT.P","CS #8", group="F. 크로스섹셔널/RV")
cs9  = input.symbol("BINANCE:ARBUSDT.P","CS #9", group="F. 크로스섹셔널/RV")
cs10 = input.symbol("BINANCE:OPUSDT.P", "CS #10", group="F. 크로스섹셔널/RV")
var string[] css = array.from(cs1,cs2,cs3,cs4,cs5,cs6,cs7,cs8,cs9,cs10)
mom(sym) =>
    c  = request.security(sym, csTF, close, barmerge.gaps_off, barmerge.lookahead_off)
    cn = request.security(sym, csTF, close[csMomLen], barmerge.gaps_off, barmerge.lookahead_off)
    na(c) or na(cn) or cn==0 ? na : (c/cn - 1.0)
rv1(sym) =>
    r  = request.security(sym, csTF, math.log(close/close[1]), barmerge.gaps_off, barmerge.lookahead_off)
    ta.sum(r*r, csRVlen)
var float[] momArr = array.new_float(), rvArr = array.new_float()
array.clear(momArr), array.clear(rvArr)
float momT = na, rvT = na
for i=0 to array.size(css)-1
    s = array.get(css,i)
    m = mom(s), v = rv1(s)
    array.push(momArr, m), array.push(rvArr, v)
    if str.contains(s, "SOLUSDT.P")
        momT := m, rvT := v
rankDesc(v, arr) =>
    rk = 1
    for j=0 to array.size(arr)-1
        if not na(v) and not na(array.get(arr,j)) and array.get(arr,j) > v
            rk += 1
    rk
rankAsc(v, arr) =>
    rk = 1
    for j=0 to array.size(arr)-1
        if not na(v) and not na(array.get(arr,j)) and array.get(arr,j) < v
            rk += 1
    rk
momRankDesc = useCS ? rankDesc(momT, momArr) : na
rvRankAsc   = useCS ? rankAsc(rvT,  rvArr)   : na
csOK_Long   = not useCS or (momRankDesc <= csTopN and (not useCS_RVlow  or rvRankAsc <= rvLowTopN))
csOK_Short  = not useCS or (momRankDesc >= (array.size(css)-csBottomN+1) and (not useCS_RVhigh or (array.size(rvArr)-rvRankAsc+1) <= rvHighTopN))

// LTF RV 게이트
useRVgate  = input.bool(true, "리얼라이즈드 분산 게이트(하위TF)", group="F. 크로스섹셔널/RV")
rvTF       = input.timeframe("1", "RV 계산 하위TF", group="F. 크로스섹셔널/RV")
rvLenMin   = input.int(60,  "RV 누적 분(분)", minval=5, group="F. 크로스섹셔널/RV")
rvSMAwin   = input.int(240, "RV 기준선 평균 분(분)", minval=30, group="F. 크로스섹셔널/RV")
rvMode     = input.string("Quiet우대", "게이트 모드", options=["Quiet우대","Vol우대","둘다허용"], group="F. 크로스섹셔널/RV")
rvSum = request.security(syminfo.tickerid, rvTF,
     ta.sum(math.pow(math.log(close/close[1]), 2), rvLenMin), lookahead=barmerge.lookahead_off)
rvMA  = request.security(syminfo.tickerid, rvTF,
     ta.sma(ta.sum(math.pow(math.log(close/close[1]), 2), rvLenMin), rvSMAwin), lookahead=barmerge.lookahead_off)
rvNorm = rvMA == 0 ? na : (rvSum / rvMA)  // 1=기준, <1 Quiet, >1 Vol
rvQuiet = useRVgate ? rvNorm < 1.0 : true
rvVol   = useRVgate ? rvNorm > 1.0 : true
rvOK    = rvMode=="Quiet우대" ? rvQuiet : rvMode=="Vol우대" ? rvVol : true

// ======================[ G. 파생/흐름 ]=====================
useBTCFilter = input.bool(false, "BTC 상관/추세 필터", group="G. 파생/흐름")
btcTicker    = input.symbol("BINANCE:BTCUSDT", "BTC 스팟", group="G. 파생/흐름")
btcTF        = input.timeframe("60", "BTC TF", group="G. 파생/흐름")
corrLen      = input.int(100, "BTC 상관 길이(바)", group="G. 파생/흐름")
btcEMAlen    = input.int(50, "BTC EMA 길이", group="G. 파생/흐름")
btcClose = request.security(btcTicker, btcTF, close, barmerge.gaps_off, barmerge.lookahead_off)
btcEMA   = request.security(btcTicker, btcTF, ta.ema(close, btcEMAlen), barmerge.gaps_off, barmerge.lookahead_off)
btcUp    = btcClose > btcEMA
ret      = math.log(close/close[1])
btcRet   = math.log(btcClose/btcClose[1])
rhoBTC   = ta.correlation(ret, btcRet, corrLen)
btcOK_L  = not useBTCFilter or (btcUp and rhoBTC >= 0)
btcOK_S  = not useBTCFilter or ((not btcUp) and rhoBTC >= 0)
useFunding   = input.bool(false, "펀딩율 필터", group="G. 파생/흐름")
fundingSym   = input.symbol("", "펀딩율 심볼", group="G. 파생/흐름")
maxLongFR    = input.float(0.05, "롱 허용 최대 펀딩%(+)", group="G. 파생/흐름")
maxShortNeg  = input.float(-0.05,"숏 허용 최소 펀딩%(-)", group="G. 파생/흐름")
fr = (useFunding and str.length(str.tostring(fundingSym))>0) ? request.security(fundingSym, timeframe.period, close, barmerge.gaps_off, barmerge.lookahead_off) : na
frOK_L = not useFunding or na(fr) or (fr <= maxLongFR/100.0)
frOK_S = not useFunding or na(fr) or (fr >= maxShortNeg/100.0)
useBasis     = input.bool(false,"베이시스(Perp-Spot) 필터", group="G. 파생/흐름")
spotSym      = input.symbol("BINANCE:SOLUSDT","스팟 심볼", group="G. 파생/흐름")
spotClose = request.security(spotSym, timeframe.period, close, lookahead=barmerge.lookahead_off)
basisPct  = (close/spotClose - 1.0) * 100.0
maxLongBasis = input.float(0.6,"롱 허용 최대 베이시스%", group="G. 파생/흐름")
minShortBasis= input.float(-0.6,"숏 허용 최소 베이시스%", group="G. 파생/흐름")
basisOK_L = not useBasis or (basisPct <= maxLongBasis)
basisOK_S = not useBasis or (basisPct >= minShortBasis)
useOI      = input.bool(false, "OI 증가 동행 요구", group="G. 파생/흐름")
oiSymbol   = input.symbol("", "OI 심볼", group="G. 파생/흐름")
oi = (useOI and str.length(str.tostring(oiSymbol))>0) ? request.security(oiSymbol, timeframe.period, close, lookahead=barmerge.lookahead_off) : na
deltaOI = na(oi) ? na : (oi - oi[1])
normDeltaOI = na(deltaOI) ? na : (deltaOI / math.max(1e-9, ta.sma(math.abs(deltaOI), 50)))
minDeltaOI = input.float(0.0, "정규화 ΔOI 최소치", group="G. 파생/흐름")
oiOK = not useOI or na(normDeltaOI) or (normDeltaOI >= minDeltaOI)
useDollarVol = input.bool(false, "유동성: 최소 달러거래대금 필터", group="G. 파생/흐름")
minDollarVol = input.float(1e6, "최소 달러 거래대금", group="G. 파생/흐름")
dollarVol = close * volume
liqOK = not useDollarVol or (dollarVol >= minDollarVol)

// ======================[ I. 엔트리 조건 (기존 모듈) ]=====================
rsi       = ta.rsi(close, rsiLen)
k         = ta.sma(ta.stoch(high, low, close, stochLen), 1)
dSig      = ta.sma(k, stochSig)
stochBull = ta.crossover(k, dSig) and k < 20
stochBear = ta.crossunder(k, dSig) and k > 80
touchLower = close <= bbDn or low <= bbDn
touchUpper = close >= bbUp or high >= bbUp
enableTrend   = input.bool(true, "트렌딩 모듈 ON", group="I. 엔트리")
enableRange   = input.bool(true, "레인지 모듈 ON",  group="I. 엔트리")
enableSqueeze = input.bool(true,"Quiet→Vol 돌파 모듈 ON", group="I. 엔트리")
useAVWAPfilter = input.bool(false, "AVWAP 방향 필터 사용", group="I. 엔트리")
avwapOK_L = not useAVWAPfilter or (useAnchors and not na(anchoredVWAP) and close >= anchoredVWAP)
avwapOK_S = not useAVWAPfilter or (useAnchors and not na(anchoredVWAP) and close <= anchoredVWAP)
trendLong  = enableTrend and (ta.adx(adxLen) >= adxTrend) and trendUp   and vwapOK_Long and inSess and htfOK and avwapOK_L
trendLong  := trendLong and (ta.crossover(close, emaFast) or stochBull or (useDonchian and ta.crossover(close, ta.highest(high, donLenEff))))
trendShort = enableTrend and (ta.adx(adxLen) >= adxTrend) and trendDown and vwapOK_Short and inSess and htfOK and avwapOK_S
trendShort := trendShort and (ta.crossunder(close, emaFast) or stochBear or (useDonchian and ta.crossunder(close, ta.lowest(low, donLenEff))))
rangeLong  = enableRange and (ta.adx(adxLen) <= adxRange) and inSess and htfOK and (rsi < rsiLow)  and (touchLower or ta.crossover(rsi, rsiLow))
rangeShort = enableRange and (ta.adx(adxLen) <= adxRange) and inSess and htfOK and (rsi > rsiHigh) and (touchUpper or ta.crossunder(rsi, rsiHigh))
squeezeLong  = enableSqueeze and vwapOK_Long  and inSess and htfOK and (not useQuiet2Vol or (isVolNow and isQuiet[1] and close > bbUp))
squeezeShort = enableSqueeze and vwapOK_Short and inSess and htfOK and (not useQuiet2Vol or (isVolNow and isQuiet[1] and close < bbDn))
orbOK = not useORB or isORBWindow

// ======================[ I2. Connors R3 모듈 ]=====================
groupR3 = "I2. Connors R3 (RSI 평균회귀)"
enableR3        = input.bool(true,  "R3 모듈 ON", group=groupR3)
r3_rsiLen       = input.int(2,      "RSI 기간", group=groupR3)
r3_consecDays   = input.int(3,      "연속 하락/상승 일수", minval=2, maxval=6, group=groupR3)
r3_useRSIstreak = input.bool(true,  "연속성 기준: RSI가 연속 하락/상승(OFF=종가)", group=groupR3)
r3_firstDayMaxRSI_Long  = input.float(60.0, "롱: 첫날 RSI 최대값(<)", step=0.5, group=groupR3)
r3_entryRSI_Long        = input.float(10.0, "롱 진입 RSI 임계(<)", step=0.5, group=groupR3)
r3_firstDayMinRSI_Short = input.float(60.0, "숏: 첫날 RSI 최소값(>)", step=0.5, group=groupR3)
r3_entryRSI_Short       = input.float(90.0, "숏 진입 RSI 임계(>)", step=0.5, group=groupR3)
r3_exitRSI_Long         = input.float(70.0, "롱 청산 RSI 임계(>)", step=0.5, group=groupR3)
r3_exitRSI_Short        = input.float(30.0, "숏 청산 RSI 임계(<)", step=0.5, group=groupR3)
r3_maTF         = input.timeframe("D", "장기 추세 MA TF", group=groupR3)
r3_maLen        = input.int(200, "장기 추세 MA 길이", group=groupR3)
r3_exactStreak  = input.bool(true, "연속 일수 정확히 일치만 허용(OFF=이상)", group=groupR3)
r3_timeStopBars = input.int(0,   "R3 전용 시간 스톱(바, 0=off)", group=groupR3)
r3_hardStopPct  = input.float(0.0,"R3 전용 하드 스톱%(0=off)", step=0.1, group=groupR3)
r3_exitMode     = input.string("Pro_Exit_Pack","R3 청산방식", options=["Classic_RSI","Pro_Exit_Pack"], group=groupR3)
r3_respectGates = input.bool(true, "R3도 글로벌 게이트(CS/RV/BTC/펀딩 등) 준수", group=groupR3)

rsi2 = ta.rsi(close, r3_rsiLen)
var int rsiDownCnt = 0, rsiUpCnt = 0, pxDownCnt = 0, pxUpCnt = 0
rsiDownCnt := (rsi2 < rsi2[1]) ? nz(rsiDownCnt[1]) + 1 : 0
rsiUpCnt   := (rsi2 > rsi2[1]) ? nz(rsiUpCnt[1]) + 1   : 0
pxDownCnt  := (close < close[1]) ? nz(pxDownCnt[1]) + 1 : 0
pxUpCnt    := (close > close[1]) ? nz(pxUpCnt[1]) + 1   : 0

streakDown = r3_useRSIstreak ? rsiDownCnt : pxDownCnt
streakUp   = r3_useRSIstreak ? rsiUpCnt   : pxUpCnt

r3_MA = request.security(syminfo.tickerid, r3_maTF, ta.sma(close, r3_maLen), lookahead=barmerge.lookahead_off)

firstRSI_Long_OK  = rsi2[r3_consecDays-1] < r3_firstDayMaxRSI_Long
firstRSI_Short_OK = rsi2[r3_consecDays-1] > r3_firstDayMinRSI_Short

streakOK_long  = r3_exactStreak ? (streakDown == r3_consecDays) : (streakDown >= r3_consecDays)
streakOK_short = r3_exactStreak ? (streakUp   == r3_consecDays) : (streakUp   >= r3_consecDays)

r3_long_base  = enableR3 and (close > r3_MA) and streakOK_long  and firstRSI_Long_OK  and (rsi2 <  r3_entryRSI_Long)
r3_short_base = enableR3 and (close < r3_MA) and streakOK_short and firstRSI_Short_OK and (rsi2 >  r3_entryRSI_Short)

// ======================[ J. CUSUM 이벤트 게이트 ]=====================
useCUSUM = input.bool(false, "CUSUM 변동 이벤트 게이트", group="J. CUSUM/게이트")
hCUSUM   = input.float(0.75, "CUSUM 임계값(표준화)", step=0.05, group="J. CUSUM/게이트")
retLog = math.log(close/close[1])
mu = ta.sma(retLog, 100)
sigma = ta.stdev(retLog, 100)
z = sigma>0? (retLog-mu)/sigma : 0.0
var float cpos=0, cneg=0
cpos := math.max(0, cpos + z - hCUSUM)
cneg := math.min(0, cneg + z + hCUSUM)
cusumEvent = (cpos==0 and z<0) or (cneg==0 and z>0)
cusumOK = not useCUSUM or cusumEvent

// ======================[ K. 게이트/진입 최종 ]=====================
dateOK  = inDate and (not (avoidWeekend and isWeekend)) and utcOK and embargoOK
flowOK_L= (btcOK_L and frOK_L and basisOK_L and oiOK and liqOK)
flowOK_S= (btcOK_S and frOK_S and basisOK_S and oiOK and liqOK)

enterLongBase0  = (trendLong or rangeLong or squeezeLong)
enterShortBase0 = (trendShort or rangeShort or squeezeShort)

r3GateOK_L = not r3_respectGates or (csOK_Long and rvOK and flowOK_L)
r3GateOK_S = not r3_respectGates or (csOK_Short and rvOK and flowOK_S)
enterLongBaseR3  = r3_long_base  and r3GateOK_L
enterShortBaseR3 = r3_short_base and r3GateOK_S

enterLongBase  = dateOK and orbOK and cusumOK and (enterLongBase0 or enterLongBaseR3)
enterShortBase = dateOK and orbOK and cusumOK and (enterShortBase0 or enterShortBaseR3)

enterLong  = barstate.isconfirmed and enterLongBase
enterShort = barstate.isconfirmed and enterShortBase

// ======================[ L. 리스크/사이징 + 적응 스로틀 ]=====================
groupRisk = "L. 리스크/사이징/스로틀"
riskPctPerTr = input.float(1.0, "트레이드당 위험% (자본 대비)", minval=0.05, maxval=10.0, group=groupRisk)
useVolSize   = input.bool(true,  "ATR-리스크 기반 사이징", group=groupRisk)
stopDistPctBase  = (ta.atr(atrLen) * atrTrailMult) / close * 100.0
minStopPct   = 0.01
basePct      = (useVolSize ? (riskPctPerTr / math.max(minStopPct, stopDistPctBase)) * 100.0 : default_qty_value)
basePct      := math.clamp(basePct, 0.1, 1000)
// Kelly
useKellyCap  = input.bool(true,  "Kelly 기반 포지션 상한", group=groupRisk)
kellyTradesN = input.int(40,     "Kelly 추정용 트레이드 수", minval=10, maxval=500, group=groupRisk)
kellyMult    = input.float(0.5,  "Kelly 배수(0.5=Half)", minval=0.1, maxval=1.0, group=groupRisk)
kellyMaxPct  = input.float(25,   "Kelly 최대 상한(% equity)", minval=1, maxval=100, group=groupRisk)
var float[] kellyRets = array.new_float()
var float   kEntry    = na
var int     kDir      = 0
openedNow  = (strategy.position_size != 0 and strategy.position_size[1] == 0)
closedNow  = (strategy.position_size == 0 and strategy.position_size[1] != 0)
if openedNow
    kEntry := strategy.position_avg_price
    kDir   := strategy.position_size > 0 ? 1 : -1
if closedNow and not na(kEntry) and kDir != 0
    retTrade = kDir == 1 ? (close / kEntry - 1.0) : (kEntry / close - 1.0)
    array.push(kellyRets, retTrade)
    if array.size(kellyRets) > kellyTradesN
        array.shift(kellyRets)
    kEntry := na, kDir := 0
winSum=0.0, winCnt=0, lossSum=0.0, lossCnt=0
for i=0 to array.size(kellyRets)-1
    v = array.get(kellyRets,i)
    if v>0
        winSum+=v
        winCnt+=1
    else if v<0
        lossSum+=-v
        lossCnt+=1
p = (winCnt+lossCnt)>0 ? winCnt/(winCnt+lossCnt) : na
Rwl = (winCnt>0 and lossCnt>0 and (lossSum/lossCnt)>0) ? ((winSum/winCnt)/(lossSum/lossCnt)) : na
fKelly = (not na(p) and not na(Rwl) and Rwl>0) ? (p - (1-p)/Rwl) : na
kellyCapPct = useKellyCap and not na(fKelly) ? math.clamp(100*kellyMult*fKelly, 0, kellyMaxPct) : na
useLossThrottle = input.bool(true, "손실연속 스로틀", group=groupRisk)
lossStepPct     = input.float(0.2, "손실 1회당 위험 감소율(예:0.2=20%)", minval=0.0, maxval=0.9, step=0.05, group=groupRisk)
maxThrottle     = input.float(0.6, "최대 위험 축소비율(예:0.6=60%)", minval=0.0, maxval=0.95, step=0.05, group=groupRisk)
var int lossStreak=0
if closedNow
    // 간단한 손익 추적: 마지막 거래 수익률 기준
    lastRet = array.size(kellyRets)>0 ? array.get(kellyRets, array.size(kellyRets)-1) : na
    lossStreak := (not na(lastRet) and lastRet<0) ? lossStreak+1 : 0
throttle = useLossThrottle ? math.min(maxThrottle, lossStreak*lossStepPct) : 0.0
basePct := basePct * (1.0 - throttle)
finalPct = (useKellyCap and not na(kellyCapPct)) ? math.min(basePct, kellyCapPct) : basePct

// ======================[ M. 터틀 피라미딩(선택) ]=====================
usePyramid   = input.bool(true, "터틀 피라미딩", group="M. 피라미딩/증액")
maxUnits     = input.int(4, "최대 유닛(≤4)", minval=1, maxval=4, group="M. 피라미딩/증액")
unitAdd_N    = input.float(0.5, "추가 유닛 간격(N)", step=0.1, group="M. 피라미딩/증액")
unitPct      = input.float(25, "유닛당 % of equity", minval=1, maxval=100, group="M. 피라미딩/증액")
Nlen_pyr     = input.int(20, "N(ATR) 길이", group="M. 피라미딩/증액")
Natr         = ta.atr(Nlen_pyr)
var int unitsNow=0
if openedNow
    unitsNow := 1
var float lastAddPx = na
if openedNow
    lastAddPx := strategy.position_avg_price
canAddLong  = usePyramid and strategy.position_size > 0 and unitsNow < maxUnits and close >= nz(lastAddPx) + unitAdd_N*Natr
canAddShort = usePyramid and strategy.position_size < 0 and unitsNow < maxUnits and close <= nz(lastAddPx) - unitAdd_N*Natr
if canAddLong
    strategy.entry("L.add."+str.tostring(unitsNow+1), strategy.long, qty_percent=unitPct)
    unitsNow += 1
    lastAddPx := close
if canAddShort
    strategy.entry("S.add."+str.tostring(unitsNow+1), strategy.short, qty_percent=unitPct)
    unitsNow += 1
    lastAddPx := close
if closedNow
    unitsNow := 0
    lastAddPx := na

// ======================[ N. 프로 EXIT PACK ]=====================
groupExit = "N. 프로 EXIT(실전)"
initStopMode = input.string("ATR_x", "초기 스톱 방식", options=["ATR_x","Turtle_2N","Chande-Kroll","Percent"], group=groupExit)
atrInitLen   = input.int(14, "ATR_x: ATR 길이", group=groupExit)
atrInitMult  = input.float(2.2, "ATR_x: 배수", step=0.1, group=groupExit)
N_len    = input.int(20, "Turtle N 길이(ATR)", group=groupExit)
N_mult   = input.float(2.0, "Turtle 초기스톱 배수(2N 권장)", step=0.1, group=groupExit)
ck_len1  = input.int(10, "Chande-Kroll L1", group=groupExit)
ck_len2  = input.int(20, "Chande-Kroll L2", group=groupExit)
ck_atr   = input.int(10, "Chande-Kroll ATR 길이", group=groupExit)
ck_mult  = input.float(1.5, "Chande-Kroll ATR 배수", step=0.1, group=groupExit)
pctStop  = input.float(1.5, "Percent: 초기 스톱% (예: 1.5=1.5%)", step=0.1, group=groupExit)
trail_Chandelier = input.bool(true,  "트레일링: Chandelier", group=groupExit)
chandLen         = input.int(22,     "Chandelier: ATR 길이", group=groupExit)
chandMult        = input.float(3.0,  "Chandelier: 배수", step=0.1, group=groupExit)
trail_Donchian   = input.bool(true,  "트레일링: Donchian", group=groupExit)
donTrailLen      = input.int(20,     "Donchian: 트레일 길이", group=groupExit)
trail_ATRx       = input.bool(true,  "트레일링: ATR x 배수", group=groupExit)
trail_ATRlen     = input.int(14,     "Trail ATR 길이", group=groupExit)
trail_ATRmult    = input.float(2.0,  "Trail ATR 배수", step=0.1, group=groupExit)
useBE_on_TP1     = input.bool(true,  "TP1 체결 시 BE(진입가로)", group=groupExit)
useBE_on_MFE     = input.bool(true,  "MFE가 R배수 이상이면 BE", group=groupExit)
be_MFE_R         = input.float(1.0,  "BE 트리거: MFE ≥ R", step=0.1, group=groupExit)
useTimeStop      = input.bool(true,  "시간 스톱", group=groupExit)
timeStopBars     = input.int(60,     "시간 스톱: 보유 바 수", minval=1, group=groupExit)
timeStopNeedProfit = input.bool(false, "시간 스톱은 미수익일 때만", group=groupExit)
useTPs      = input.bool(true, "TP1/TP2 분할청산", group=groupExit)
tp1_RR      = input.float(1.5, "TP1 RR", group=groupExit)
tp2_RR      = input.float(3.0, "TP2 RR", group=groupExit)
tp1_pct     = input.float(50, "TP1 % 청산", minval=1, maxval=99, group=groupExit)

// === 진입 ===
var int r3PosTag = 0  // 1=R3롱, -1=R3숏, 0=기타
r3EnterLongNow  = enableR3 and enterLongBaseR3 and (strategy.position_size <= 0)
r3EnterShortNow = enableR3 and enterShortBaseR3 and (strategy.position_size >= 0)

if enterLong and strategy.position_size <= 0
    strategy.entry("L", strategy.long,  qty_percent=finalPct)
    r3PosTag := r3EnterLongNow ? 1 : 0
if enterShort and strategy.position_size >= 0
    strategy.entry("S", strategy.short, qty_percent=finalPct)
    r3PosTag := r3EnterShortNow ? -1 : 0
if closedNow
    r3PosTag := 0

// === 초기 스톱 계산 ===
atrInit = ta.atr(atrInitLen)
N      = ta.atr(N_len)
ckATR  = ta.atr(ck_atr)
var float initStopL = na, initStopS = na, entryPx = na
openedNow2  = (strategy.position_size != 0 and strategy.position_size[1] == 0)
if openedNow2
    entryPx := strategy.position_avg_price
    if strategy.position_size > 0
        initStopL := switch initStopMode
            "ATR_x"        => entryPx - atrInitMult * atrInit
            "Turtle_2N"    => entryPx - N_mult * N
            "Chande-Kroll" => math.max(ta.lowest(low, ck_len2), ta.lowest(low, ck_len1)) - ck_mult * ckATR
            => entryPx * (1.0 - pctStop/100.0)
        initStopS := na
    else if strategy.position_size < 0
        initStopS := switch initStopMode
            "ATR_x"        => entryPx + atrInitMult * atrInit
            "Turtle_2N"    => entryPx + N_mult * N
            "Chande-Kroll" => math.min(ta.highest(high, ck_len2), ta.highest(high, ck_len1)) + ck_mult * ckATR
            => entryPx * (1.0 + pctStop/100.0)
        initStopL := na

// === MFE/BE 계산을 위한 R ===
var float riskPerTrade = na
if openedNow2
    riskPerTrade := strategy.position_size > 0 ? (entryPx - initStopL) : (initStopS - entryPx)
riskPerTrade := na(riskPerTrade) or riskPerTrade<=0 ? ta.atr(atrLen)*atrTrailMult : riskPerTrade

// Track HH/LL since entry for MFE and Chandelier
var float hhSinceEntry = na
var float llSinceEntry = na
if openedNow2
    hhSinceEntry := high
    llSinceEntry := low
else
    if strategy.position_size > 0
        hhSinceEntry := na(hhSinceEntry) ? high : math.max(hhSinceEntry, high)
    if strategy.position_size < 0
        llSinceEntry := na(llSinceEntry) ? low  : math.min(llSinceEntry, low)

longMFE  = strategy.position_size > 0 and not na(entryPx) and not na(hhSinceEntry) ? (hhSinceEntry - entryPx) : na
shortMFE = strategy.position_size < 0 and not na(entryPx) and not na(llSinceEntry) ? (entryPx - llSinceEntry) : na
mfeR = strategy.position_size > 0 ? (longMFE / math.max(1e-9, riskPerTrade)) :
       strategy.position_size < 0 ? (shortMFE/ math.max(1e-9, riskPerTrade)) : na

// === 트레일링 후보들 ===
chATR = ta.atr(chandLen)
chL = strategy.position_size > 0 and not na(hhSinceEntry) ? hhSinceEntry - chandMult * chATR : na
chS = strategy.position_size < 0 and not na(llSinceEntry) ? llSinceEntry + chandMult * chATR : na
donTrailL = strategy.position_size > 0 ? ta.lowest(low, donTrailLen) : na
donTrailS = strategy.position_size < 0 ? ta.highest(high, donTrailLen) : na
tATR = ta.atr(trail_ATRlen)
atrTrailL = strategy.position_size > 0 ? (close - trail_ATRmult * tATR) : na
atrTrailS = strategy.position_size < 0 ? (close + trail_ATRmult * tATR) : na

beTriggered = false
if strategy.position_size > 0
    beTriggered := (useBE_on_TP1 and close >= entryPx + riskPerTrade*tp1_RR) or (useBE_on_MFE and not na(mfeR) and mfeR >= be_MFE_R)
if strategy.position_size < 0
    beTriggered := (useBE_on_TP1 and close <= entryPx - riskPerTrade*tp1_RR) or (useBE_on_MFE and not na(mfeR) and mfeR >= be_MFE_R)
bePriceL = entryPx
bePriceS = entryPx

var float finalStopL = na
var float finalStopS = na

if strategy.position_size > 0
    float s1 = na(initStopL) ? -1e10 : initStopL
    float s2 = (trail_Chandelier and not na(chL)) ? chL : -1e10
    float s3 = (trail_Donchian   and not na(donTrailL)) ? donTrailL : -1e10
    float s4 = (trail_ATRx       and not na(atrTrailL)) ? atrTrailL : -1e10
    float s5 = beTriggered ? bePriceL : -1e10
    finalStopL := math.max(math.max(s1, s2), math.max(math.max(s3, s4), s5))
else
    finalStopL := na

if strategy.position_size < 0
    float t1 = na(initStopS) ? 1e10 : initStopS
    float t2 = (trail_Chandelier and not na(chS)) ? chS : 1e10
    float t3 = (trail_Donchian   and not na(donTrailS)) ? donTrailS : 1e10
    float t4 = (trail_ATRx       and not na(atrTrailS)) ? atrTrailS : 1e10
    float t5 = beTriggered ? bePriceS : 1e10
    finalStopS := math.min(math.min(t1, t2), math.min(math.min(t3, t4), t5))
else
    finalStopS := na

tp1L  = strategy.position_size > 0 ? entryPx + riskPerTrade * tp1_RR : na
tp2L  = strategy.position_size > 0 ? entryPx + riskPerTrade * tp2_RR : na
tp1S  = strategy.position_size < 0 ? entryPx - riskPerTrade * tp1_RR : na
tp2S  = strategy.position_size < 0 ? entryPx - riskPerTrade * tp2_RR : na

// 시간 스톱/보유 제한(공용)
var int barsInPos = 0
barsInPos := strategy.position_size != 0 ? nz(barsInPos) + 1 : 0
timeStopHit = false
if useTimeStop and strategy.position_size != 0
    unreal = strategy.position_size > 0 ? (close - entryPx) : (entryPx - close)
    timeStopHit := (barsInPos >= timeStopBars) and (not timeStopNeedProfit or unreal <= 0)

// === 주문/청산(공용) ===
if strategy.position_size > 0
    strategy.exit("XL_TP1", from_entry="L", stop = timeStopHit ? close : finalStopL, limit= useTPs ? tp1L : na, qty_percent= useTPs ? tp1_pct : 100)
    if useTPs
        strategy.exit("XL_TP2", from_entry="L", stop = timeStopHit ? close : finalStopL, limit=tp2L, qty_percent=100 - tp1_pct)
if strategy.position_size < 0
    strategy.exit("XS_TP1", from_entry="S", stop = timeStopHit ? close : finalStopS, limit= useTPs ? tp1S : na, qty_percent= useTPs ? tp1_pct : 100)
    if useTPs
        strategy.exit("XS_TP2", from_entry="S", stop = timeStopHit ? close : finalStopS, limit=tp2S, qty_percent=100 - tp1_pct)

// === R3 전용 청산 (Classic 모드 선택 시, R3 태그 포지션에만 적용) ===
r3ExitLong = enableR3 and (r3_exitMode=="Classic_RSI") and (r3PosTag==1) and (rsi2 > r3_exitRSI_Long)
r3ExitShort= enableR3 and (r3_exitMode=="Classic_RSI") and (r3PosTag==-1) and (rsi2 < r3_exitRSI_Short)
if r3ExitLong and strategy.position_size > 0
    strategy.close(id="L", comment="R3 RSI Exit L")
if r3ExitShort and strategy.position_size < 0
    strategy.close(id="S", comment="R3 RSI Exit S")

// === R3 전용 시간/하드 스톱(선택) ===
if enableR3 and r3PosTag!=0 and strategy.position_size!=0
    if r3_timeStopBars>0 and barsInPos>=r3_timeStopBars
        strategy.close_all(comment="R3 Time Stop")
    if r3_hardStopPct>0
        if r3PosTag==1 and (close <= entryPx*(1.0 - r3_hardStopPct/100.0))
            strategy.close_all(comment="R3 Hard Stop L")
        if r3PosTag==-1 and (close >= entryPx*(1.0 + r3_hardStopPct/100.0))
            strategy.close_all(comment="R3 Hard Stop S")

// 보유시간 하드 캡(옵션)
if (maxBarsHold > 0 and barsInPos >= maxBarsHold) and strategy.position_size != 0
    strategy.close_all(comment="Time Exit MaxHold")
    barsInPos := 0

// ======================[ O. 데일리 손실캡 & 드로우다운 킬스위치 ]=====================
groupDD = "O. 리스크 캡/쿨다운"
useDailyCap   = input.bool(true,  "일일 손실 캡/중지", group=groupDD)
capPct        = input.float(3.0,  "일손실 캡 % (자본 대비)", step=0.1, group=groupDD)
cooldownBars  = input.int(60,     "캡 히트 후 쿨다운 바", minval=0, group=groupDD)
var float dayStartEq = na
var int   cooldown=0
newDayTV = ta.change(time("D"))
if newDayTV
    dayStartEq := strategy.equity
    cooldown := 0
dayStartEq := na(dayStartEq) ? strategy.equity : dayStartEq
todayPnLPct = (strategy.equity - dayStartEq) / dayStartEq * 100.0
capHit = useDailyCap and todayPnLPct <= -capPct
cooldown := capHit ? cooldownBars : math.max(0, cooldown - 1)
useKillDD = input.bool(true, "MDD 킬스위치", group=groupDD)
killPct   = input.float(20.0, "최대낙폭 허용 %", step=0.5, group=groupDD)
var float eqHigh = na
eqHigh := na(eqHigh) ? strategy.equity : math.max(eqHigh, strategy.equity)
mddPct = (strategy.equity - eqHigh) / eqHigh * 100.0
killHit = useKillDD and (mddPct <= -killPct)

entryGateOK = not capHit and cooldown==0 and not killHit
enterLong  := enterLong  and entryGateOK
enterShort := enterShort and entryGateOK

// =========================[ P. 시각화 ]========================
plot(emaFast, "EMA Fast", color=color.new(color.teal, 0))
plot(emaSlow, "EMA Slow", color=color.new(color.orange, 0))
plot(bbUp,   "BB Upper",  color=color.new(color.gray, 50))
plot(bbDn,   "BB Lower",  color=color.new(color.gray, 50))
plot(stLine, "Supertrend", color=color.new(color.fuchsia, 0), display=(trendFilter=="Supertrend")? display.all : display.none)
plot(kama,   "KAMA",       color=color.new(color.yellow, 0),  display=(trendFilter=="KAMA")?       display.all : display.none)
plot(sessVWAP, "세션 VWAP", color=color.new(color.blue,60), display=useAVWAP?display.all:display.none)
plot(anchoredVWAP, "Anchored VWAP", color=color.new(color.aqua,0), display=useAnchors?display.all:display.none)
plot(r3_MA, "R3 200MA(TF)", color=color.new(color.purple, 0), display=enableR3?display.all:display.none)
bgcolor( (ta.adx(adxLen)>=adxTrend and trendUp) ? color.new(color.green,85) :
       (ta.adx(adxLen)<=adxRange) ? color.new(color.blue,85)  :
       isQuiet ? color.new(color.gray,88)  :
       isVolNow? color.new(color.red,88)   : na)

// ======================[ Q. 알림(상시) ]=====================
msgLong  = '{"side":"buy","symbol":"{{ticker}}","price":"{{close}}","id":"KASIA_RA_v5_2_R3","ts":"{{timenow}}"}'
msgShort = '{"side":"sell","symbol":"{{ticker}}","price":"{{close}}","id":"KASIA_RA_v5_2_R3","ts":"{{timenow}}"}'
alertcondition(enterLong,  title="Long Entry",  message=msgLong)
alertcondition(enterShort, title="Short Entry", message=msgShort)
alertcondition(strategy.position_size>0 and not na(finalStopL) and close <= finalStopL, title="Long Stop",  message='{"side":"close_long","symbol":"{{ticker}}","price":"{{close}}","id":"KASIA_RA_v5_2_R3"}')
alertcondition(strategy.position_size<0 and not na(finalStopS) and close >= finalStopS, title="Short Stop", message='{"side":"close_short","symbol":"{{ticker}}","price":"{{close}}","id":"KASIA_RA_v5_2_R3"}')
alertcondition(enableR3 and (r3ExitLong or r3ExitShort), title="R3 Classic Exit", message='{"side":"close","reason":"R3_RSI","symbol":"{{ticker}}","price":"{{close}}","id":"KASIA_RA_v5_2_R3"}')//@version=5
strategy(
     title = "KASIA Regime",
     overlay = true,
     max_bars_back = 5000,

     // === 자본/수수료/슬리피지 ===
     initial_capital  = input.float(10000, "초기자본(USD)", minval = 1, group = "A. 기본/운영"),
     commission_type  = strategy.commission.percent,
     commission_value = input.bool(true, "수수료 0.05% 적용?", group = "A. 기본/운영") ? 0.05 : 0.0,
     slippage         = input.int(0, "슬리피지(틱)", minval = 0, group = "A. 기본/운영"),

     // === 기본 주문 크기(%) ===
     default_qty_type  = strategy.percent_of_equity,
     default_qty_value = input.float(100, "기본 주문 크기(% of equity)", minval = 0.1, maxval = 10000, group = "A. 기본/운영"),

     // === 레버리지(마진%) ===  // margin_% = 100 / leverage_x
     margin_long  = input.bool(true, "레버리지(롱) 사용?", group = "A. 기본/운영")
                     ? (100.0 / input.float(5.0, "롱 레버리지(x)",  minval = 1.0, group = "A. 기본/운영"))
                     : 100.0,
     margin_short = input.bool(true, "레버리지(숏) 사용?", group = "A. 기본/운영")
                     ? (100.0 / input.float(5.0, "숏 레버리지(x)", minval = 1.0, group = "A. 기본/운영"))
                     : 100.0,

     pyramiding = 4,
     calc_on_every_tick = false,
     process_orders_on_close = true
     )   

// =========================[ A. 심볼/기본/날짜 ]========================
symBase      = input.symbol("BINANCE:SOLUSDT.P", "기본 심볼", group="A. 기본/운영")
maxBarsHold  = input.int(0, "최대 보유 바 수(0=무제한)", minval=0, group="A. 기본/운영")
// 백테스트 시작일 (시작일만)
useStart = input.bool(true, "백테스트 시작일 사용?", group="A. 기본/운영")
startAt  = input.time(timestamp("2024-01-01 00:00 +0000"), "시작일(UTC)", group="A. 기본/운영")
inDate   = not useStart or (time >= startAt)

// 경고 라벨
isSOLP = str.contains(syminfo.ticker, "SOLUSDT.P") and str.contains(syminfo.ticker, "BINANCE")
if not isSOLP
    label.new(bar_index, high, "경고: 기본은 BINANCE:SOLUSDT.P.\n다른 심볼이면 파라미터 재점검 권장.", style=label.style_label_down, textcolor=color.white, color=color.new(color.red, 0))

// ======================[ B. UTC/상위봉/세션 ]=====================
useUTCgate   = input.bool(false, "UTC 진입 제한 사용", group="B. UTC/상위봉/세션")
utcStartH    = input.int(7,  "UTC 시작-시(0~23)", minval=0, maxval=23, group="B. UTC/상위봉/세션")
utcStartM    = input.int(0,  "UTC 시작-분(0~59)", minval=0, maxval=59, group="B. UTC/상위봉/세션")
utcEndH      = input.int(21, "UTC 종료-시(0~23)", minval=0, maxval=23, group="B. UTC/상위봉/세션")
utcEndM      = input.int(0,  "UTC 종료-분(0~59)", minval=0, maxval=59, group="B. UTC/상위봉/세션")
utcNowMin = hour(time, "UTC")*60 + minute(time, "UTC")
utcStartMin = utcStartH*60 + utcStartM
utcEndMin   = utcEndH*60 + utcEndM
utcOK = not useUTCgate or (utcStartMin <= utcEndMin ? (utcNowMin >= utcStartMin and utcNowMin <= utcEndMin)
                                                 : (utcNowMin >= utcStartMin or  utcNowMin <= utcEndMin))

useFundingEmbargo = input.bool(false, "펀딩 롤오버 금지창(00/08/16 UTC ±N분)", group="B. UTC/상위봉/세션")
embargoMin        = input.int(5, "금지창 분(±)", minval=0, maxval=30, group="B. UTC/상위봉/세션")
fundHours = array.from(0, 8, 16)
hUTC = hour(time, "UTC"), mUTC = minute(time, "UTC")
isFundHour = array.includes(fundHours, hUTC)
embargoOK = not useFundingEmbargo or not (isFundHour and (mUTC <= embargoMin or mUTC >= 60-embargoMin))

useMTF      = input.bool(true, "상위봉 필터 사용(EMA+RSI)", group="B. UTC/상위봉/세션")
tf_htf      = input.timeframe("60", "상위봉 TF (예: 60/240)", group="B. UTC/상위봉/세션")
sym     = syminfo.tickerid
htfEMA  = request.security(sym, tf_htf, ta.ema(close, 50),  barmerge.gaps_off, barmerge.lookahead_off)
htfRSI  = request.security(sym, tf_htf, ta.rsi(close, 14),   barmerge.gaps_off, barmerge.lookahead_off)
htfOK   = not useMTF or (barstate.isconfirmed and close > htfEMA and htfEMA > nz(htfEMA[1]) and htfRSI >= 50)

useSessions  = input.bool(false, "세션 필터(아시아/런던/뉴욕)", group="B. UTC/상위봉/세션")
asiaSess     = input.session("0000-0900", "아시아(차트TZ)", group="B. UTC/상위봉/세션")
lonSess      = input.session("0700-1600", "런던(차트TZ)", group="B. UTC/상위봉/세션")
nySess       = input.session("1300-2200", "뉴욕(차트TZ)", group="B. UTC/상위봉/세션")
avoidWeekend = input.bool(false, "주말 진입 회피", group="B. UTC/상위봉/세션")
inAsia = not useSessions or not na(time(timeframe.period, asiaSess))
inLon  = not useSessions or not na(time(timeframe.period, lonSess))
inNY   = not useSessions or not na(time(timeframe.period, nySess))
inSess = not useSessions or (inAsia or inLon or inNY)
isWeekend = (dayofweek == dayofweek.saturday or dayofweek == dayofweek.sunday)

// ======================[ C. 오실레이터/AVWAP/Anchors ]=====================
useAVWAP     = input.bool(false, "세션 VWAP 필터", group="C. 오실/AVWAP/앵커")
rsiLen       = input.int(7, "RSI 기간(레인지)", group="C. 오실/AVWAP/앵커")
stochLen     = input.int(14, "스토캐스틱 K 기간", group="C. 오실/AVWAP/앵커")
stochSig     = input.int(3,  "스토캐스틱 D 기간", group="C. 오실/AVWAP/앵커")
sessVWAP = ta.vwap
vwapOK_Long  = not useAVWAP or close >= sessVWAP
vwapOK_Short = not useAVWAP or close <= sessVWAP

// Anchored VWAP (일/주/월/커스텀 앵커)
useAnchors = input.bool(false, "Anchored VWAP(AVWAP) 사용", group="C. 오실/AVWAP/앵커")
anchorDaily = input.bool(true, "일봉 시작 앵커", group="C. 오실/AVWAP/앵커")
anchorWeekly= input.bool(false,"주봉 시작 앵커", group="C. 오실/AVWAP/앵커")
anchorMonthly=input.bool(false,"월봉 시작 앵커", group="C. 오실/AVWAP/앵커")
useCustomAnchor = input.bool(false, "커스텀 앵커 사용", group="C. 오실/AVWAP/앵커")
customAnchorAt  = input.time(timestamp("2024-01-01 00:00 +0000"), "커스텀 앵커(UTC)", group="C. 오실/AVWAP/앵커")
var float num = na
var float den = na
newDay  = ta.change(time("D"))
newWeek = ta.change(time("W"))
newMonth= ta.change(time("M"))
resetAVWAP = (useAnchors and ((anchorDaily and newDay) or (anchorWeekly and newWeek) or (anchorMonthly and newMonth))) or (useCustomAnchor and time == customAnchorAt)
if na(num) or resetAVWAP
    num := hlc3*volume
    den := volume
else
    num += hlc3*volume
    den += volume
anchoredVWAP = useAnchors ? (den>0? num/den : na) : na

// ======================[ H. WFA 파라미터 뱅크 ]=====================
useWFA    = input.bool(true, "WFA 파라미터 뱅크", group="H. WFA")
wfaDays   = input.int(30, "세트 기간(일)", minval=5, group="H. WFA")
wfaSets   = input.int(3, "세트 개수", minval=1, maxval=5, group="H. WFA")
var float[] A_trend = array.from(26.0, 24.0, 28.0)
var float[] A_range = array.from(18.0, 20.0, 16.0)
var float[] A_bbMul = array.from(2.0,  2.2,  1.8)
var float[] A_qMult = array.from(0.90, 0.85, 0.95)
var float[] A_vMult = array.from(1.30, 1.60, 1.40)
var float[] A_atrTm = array.from(2.2,  1.8,  2.8)
var float[] A_emaF  = array.from(20.0, 13.0, 34.0)
var float[] A_emaS  = array.from(50.0, 55.0, 89.0)
var float[] A_rsiL  = array.from(25.0, 30.0, 28.0)
var float[] A_rsiH  = array.from(75.0, 70.0, 72.0)
var float[] A_don   = array.from(20.0, 30.0, 14.0)
wfaIdx() =>
    if not useWFA
        0
    else
        winMS = int(wfaDays) * 24 * 60 * 60 * 1000
        seg   = int(math.floor((time - startAt) / winMS))
        math.max(0, seg % wfaSets)
idx = wfaIdx()

// ======================[ D. 레짐/트렌드 + 히스테리시스 ]=====================
adxLen      = input.int(14, "ADX 기간", group="D. 레짐/트렌드/히스테리시스")
adxTrend_i  = input.float(25.0, "추세장: ADX ≥", group="D. 레짐/트렌드/히스테리시스")
adxRange_i  = input.float(18.0, "레인지: ADX ≤", group="D. 레짐/트렌드/히스테리시스")
emaFastLen_i= input.int(20, "EMA Fast", group="D. 레짐/트렌드/히스테리시스")
emaSlowLen_i= input.int(50, "EMA Slow", group="D. 레짐/트렌드/히스테리시스")
trendFilter  = input.string("EMA", "주 추세필터", options=["EMA","Supertrend","KAMA"], group="D. 레짐/트렌드/히스테리시스")
stFactor     = input.float(2.0, "Supertrend ATR factor", group="D. 레짐/트렌드/히스테리시스")
stPeriod     = input.int(10,   "Supertrend ATR period", group="D. 레짐/트렌드/히스테리시스")
kamaLen      = input.int(10,   "KAMA length", group="D. 레짐/트렌드/히스테리시스")

// Effective params from WFA
adxTrend     = useWFA ? array.get(A_trend, idx) : adxTrend_i
adxRange     = useWFA ? array.get(A_range, idx) : adxRange_i
bbLenEff     = input.int(20, "BB 길이", group="E. 스퀴즈/NR/돈키안/ORB/캔들")
bbMultEff    = useWFA ? array.get(A_bbMul, idx) : input.float(2.0, "BB 배수", step=0.1, group="E. 스퀴즈/NR/돈키안/ORB/캔들")
quietMult    = useWFA ? array.get(A_qMult, idx) : input.float(0.80, "Quiet if BBW < MA*quietMult", group="E. 스퀴즈/NR/돈키안/ORB/캔들")
volatileMult = useWFA ? array.get(A_vMult, idx) : input.float(1.50, "Volatile if BBW > MA*volatileMult", group="E. 스퀴즈/NR/돈키안/ORB/캔들")
atrTrailMult = useWFA ? array.get(A_atrTm, idx) : 2.0
emaFastLen   = useWFA ? int(array.get(A_emaF, idx)) : emaFastLen_i
emaSlowLen   = useWFA ? int(array.get(A_emaS, idx)) : emaSlowLen_i
rsiLow       = useWFA ? int(array.get(A_rsiL, idx)) : 30
rsiHigh      = useWFA ? int(array.get(A_rsiH, idx)) : 70
donLenEff    = useWFA ? int(array.get(A_don,  idx)) : 20

emaFast   = ta.ema(close, emaFastLen)
emaSlow   = ta.ema(close, emaSlowLen)
atrLen    = input.int(14, "ATR 기간", group="D. 레짐/트렌드/히스테리시스")
atrVal    = ta.atr(atrLen)

float stLine = na
int   stDir  = na
if trendFilter == "Supertrend"
    [stLine, stDir] := ta.supertrend(stFactor, stPeriod)
kama = trendFilter == "KAMA" ? ta.kama(close, kamaLen) : na
trendUp_now   = trendFilter == "EMA"        ? (close > emaSlow and emaFast > emaSlow) :
                trendFilter == "Supertrend" ? (stDir == -1) :
                (close > kama)
trendDown_now = trendFilter == "EMA"        ? (close < emaSlow and emaFast < emaSlow) :
                trendFilter == "Supertrend" ? (stDir == 1) :
                (close < kama)

// 히스테리시스: 상태 전환에 확증 바 요구
confirmBars = input.int(2, "레짐 전환 확증 바 수", minval=0, group="D. 레짐/트렌드/히스테리시스")
var int trendUpScore=0, trendDownScore=0
trendUpScore   := trendUp_now   ? math.min(confirmBars, trendUpScore+1)   : 0
trendDownScore := trendDown_now ? math.min(confirmBars, trendDownScore+1) : 0
trendUp   = trendUpScore   >= confirmBars
trendDown = trendDownScore >= confirmBars

// ======================[ E. 스퀴즈/NR/돈키안/ORB/캔들 ]=====================
groupTrig = "E. 스퀴즈/NR/돈키안/ORB/캔들"
bbwLookback = input.int(50, "BBW 평균 기간", group=groupTrig)
kcLen        = input.int(20,     "Keltner EMA 길이", group=groupTrig)
kcMult       = input.float(1.5,  "Keltner ATR 배수", group=groupTrig)
useTTMSq     = input.bool(true,  "TTM Squeeze 확인(BB in KC)", group=groupTrig)
useQuiet2Vol = input.bool(true,  "Quiet→Vol 전환 요구", group=groupTrig)
useNR   = input.bool(true, "NR7/IDNR4", group=groupTrig)
useORB  = input.bool(false,"오프닝 레인지 브레이크(UTC)", group=groupTrig)
orbMin  = input.int(15, "오프닝 범위 분", minval=1, group=groupTrig)
useCandles = input.bool(true, "캔들(엔걸핑/핀바)", group=groupTrig)
useDonchian  = input.bool(true, "돈키안 돌파(추세장)", group=groupTrig)

bbLen = bbLenEff
bbMid     = ta.sma(close, bbLen)
bbUp      = bbMid + bbMultEff * ta.stdev(close, bbLen)
bbDn      = bbMid - bbMultEff * ta.stdev(close, bbLen)
bbw       = (bbUp - bbDn) / bbMid
bbwMA     = ta.sma(bbw, bbwLookback)
isQuiet   = bbw < bbwMA * quietMult
isVolNow  = bbw > bbwMA * volatileMult
kcMid   = ta.ema(close, kcLen)
kcUp    = kcMid + kcMult * atrVal
kcDn    = kcMid - kcMult * atrVal
ttmSqueezeOn = (bbUp < kcUp and bbDn > kcDn)
rng(b) => high[b] - low[b]
isNR7   = useNR and (rng(0) <= ta.lowest(rng(0), 7))
isInside= useNR and (high <= high[1] and low >= low[1])
isNR4   = useNR and (rng(0) <= ta.lowest(rng(0), 4))
isIDNR4 = useNR and isInside and isNR4
dayStartUTC = timestamp("UTC", year, month, dayofmonth, 0, 0)
isORBWindow = useORB ? (time - dayStartUTC) <= orbMin * 60 * 1000 : true
donLen = donLenEff
donU = ta.highest(high, donLen)
donL = ta.lowest(low,  donLen)
body   = math.abs(close-open)
upperW = high - math.max(close, open)
lowerW = math.min(close, open) - low
bullEng = useCandles and (close>open and close[1]<open[1] and close>=open[1] and open<=close[1])
bearEng = useCandles and (close<open and close[1]>open[1] and close<=open[1] and open>=close[1])
hammer   = useCandles and (lowerW >= body*2 and upperW <= body and close>open)
shooting = useCandles and (upperW >= body*2 and lowerW <= body and close<open)
patternBull = bullEng or hammer
patternBear = bearEng or shooting
q2vUp   = isVolNow and isQuiet[1]  and close > bbUp
q2vDown = isVolNow and isQuiet[1]  and close < bbDn

// ======================[ F. 크로스섹셔널 + RV 게이트 ]=====================
useCS        = input.bool(true, "크로스섹셔널 필터", group="F. 크로스섹셔널/RV")
csTF         = input.timeframe("60", "CS TF", group="F. 크로스섹셔널/RV")
csMomLen     = input.int(24, "모멘텀 Lookback(바)", group="F. 크로스섹셔널/RV")
csRVlen      = input.int(96, "리얼라이즈드 분산 윈도우(바)", group="F. 크로스섹셔널/RV")
csTopN       = input.int(4,  "롱: 모멘텀 상위 N", minval=1, group="F. 크로스섹셔널/RV")
csBottomN    = input.int(4,  "숏: 모멘텀 하위 N", minval=1, group="F. 크로스섹셔널/RV")
useCS_RVlow  = input.bool(true,  "롱: 저분산 선호", group="F. 크로스섹셔널/RV")
rvLowTopN    = input.int(5, "롱: 저분산 상위 N 컷", minval=1, group="F. 크로스섹셔널/RV")
useCS_RVhigh = input.bool(false, "숏: 고분산 선호", group="F. 크로스섹셔널/RV")
rvHighTopN   = input.int(5, "숏: 고분산 상위 N 컷", minval=1, group="F. 크로스섹셔널/RV")
cs1  = input.symbol("BINANCE:BTCUSDT.P", "CS #1", group="F. 크로스섹셔널/RV")
cs2  = input.symbol("BINANCE:ETHUSDT.P", "CS #2", group="F. 크로스섹셔널/RV")
cs3  = input.symbol("BINANCE:SOLUSDT.P", "CS #3(타깃)", group="F. 크로스섹셔널/RV")
cs4  = input.symbol("BINANCE:BNBUSDT.P", "CS #4", group="F. 크로스섹셔널/RV")
cs5  = input.symbol("BINANCE:XRPUSDT.P", "CS #5", group="F. 크로스섹셔널/RV")
cs6  = input.symbol("BINANCE:ADAUSDT.P", "CS #6", group="F. 크로스섹셔널/RV")
cs7  = input.symbol("BINANCE:DOGEUSDT.P","CS #7", group="F. 크로스섹셔널/RV")
cs8  = input.symbol("BINANCE:LINKUSDT.P","CS #8", group="F. 크로스섹셔널/RV")
cs9  = input.symbol("BINANCE:ARBUSDT.P","CS #9", group="F. 크로스섹셔널/RV")
cs10 = input.symbol("BINANCE:OPUSDT.P", "CS #10", group="F. 크로스섹셔널/RV")
var string[] css = array.from(cs1,cs2,cs3,cs4,cs5,cs6,cs7,cs8,cs9,cs10)
mom(sym) =>
    c  = request.security(sym, csTF, close, barmerge.gaps_off, barmerge.lookahead_off)
    cn = request.security(sym, csTF, close[csMomLen], barmerge.gaps_off, barmerge.lookahead_off)
    na(c) or na(cn) or cn==0 ? na : (c/cn - 1.0)
rv1(sym) =>
    r  = request.security(sym, csTF, math.log(close/close[1]), barmerge.gaps_off, barmerge.lookahead_off)
    ta.sum(r*r, csRVlen)
var float[] momArr = array.new_float(), rvArr = array.new_float()
array.clear(momArr), array.clear(rvArr)
float momT = na, rvT = na
for i=0 to array.size(css)-1
    s = array.get(css,i)
    m = mom(s), v = rv1(s)
    array.push(momArr, m), array.push(rvArr, v)
    if str.contains(s, "SOLUSDT.P")
        momT := m, rvT := v
rankDesc(v, arr) =>
    rk = 1
    for j=0 to array.size(arr)-1
        if not na(v) and not na(array.get(arr,j)) and array.get(arr,j) > v
            rk += 1
    rk
rankAsc(v, arr) =>
    rk = 1
    for j=0 to array.size(arr)-1
        if not na(v) and not na(array.get(arr,j)) and array.get(arr,j) < v
            rk += 1
    rk
momRankDesc = useCS ? rankDesc(momT, momArr) : na
rvRankAsc   = useCS ? rankAsc(rvT,  rvArr)   : na
csOK_Long   = not useCS or (momRankDesc <= csTopN and (not useCS_RVlow  or rvRankAsc <= rvLowTopN))
csOK_Short  = not useCS or (momRankDesc >= (array.size(css)-csBottomN+1) and (not useCS_RVhigh or (array.size(rvArr)-rvRankAsc+1) <= rvHighTopN))

// LTF RV 게이트
useRVgate  = input.bool(true, "리얼라이즈드 분산 게이트(하위TF)", group="F. 크로스섹셔널/RV")
rvTF       = input.timeframe("1", "RV 계산 하위TF", group="F. 크로스섹셔널/RV")
rvLenMin   = input.int(60,  "RV 누적 분(분)", minval=5, group="F. 크로스섹셔널/RV")
rvSMAwin   = input.int(240, "RV 기준선 평균 분(분)", minval=30, group="F. 크로스섹셔널/RV")
rvMode     = input.string("Quiet우대", "게이트 모드", options=["Quiet우대","Vol우대","둘다허용"], group="F. 크로스섹셔널/RV")
rvSum = request.security(syminfo.tickerid, rvTF,
     ta.sum(math.pow(math.log(close/close[1]), 2), rvLenMin), lookahead=barmerge.lookahead_off)
rvMA  = request.security(syminfo.tickerid, rvTF,
     ta.sma(ta.sum(math.pow(math.log(close/close[1]), 2), rvLenMin), rvSMAwin), lookahead=barmerge.lookahead_off)
rvNorm = rvMA == 0 ? na : (rvSum / rvMA)  // 1=기준, <1 Quiet, >1 Vol
rvQuiet = useRVgate ? rvNorm < 1.0 : true
rvVol   = useRVgate ? rvNorm > 1.0 : true
rvOK    = rvMode=="Quiet우대" ? rvQuiet : rvMode=="Vol우대" ? rvVol : true

// ======================[ G. 파생/흐름 ]=====================
useBTCFilter = input.bool(false, "BTC 상관/추세 필터", group="G. 파생/흐름")
btcTicker    = input.symbol("BINANCE:BTCUSDT", "BTC 스팟", group="G. 파생/흐름")
btcTF        = input.timeframe("60", "BTC TF", group="G. 파생/흐름")
corrLen      = input.int(100, "BTC 상관 길이(바)", group="G. 파생/흐름")
btcEMAlen    = input.int(50, "BTC EMA 길이", group="G. 파생/흐름")
btcClose = request.security(btcTicker, btcTF, close, barmerge.gaps_off, barmerge.lookahead_off)
btcEMA   = request.security(btcTicker, btcTF, ta.ema(close, btcEMAlen), barmerge.gaps_off, barmerge.lookahead_off)
btcUp    = btcClose > btcEMA
ret      = math.log(close/close[1])
btcRet   = math.log(btcClose/btcClose[1])
rhoBTC   = ta.correlation(ret, btcRet, corrLen)
btcOK_L  = not useBTCFilter or (btcUp and rhoBTC >= 0)
btcOK_S  = not useBTCFilter or ((not btcUp) and rhoBTC >= 0)
useFunding   = input.bool(false, "펀딩율 필터", group="G. 파생/흐름")
fundingSym   = input.symbol("", "펀딩율 심볼", group="G. 파생/흐름")
maxLongFR    = input.float(0.05, "롱 허용 최대 펀딩%(+)", group="G. 파생/흐름")
maxShortNeg  = input.float(-0.05,"숏 허용 최소 펀딩%(-)", group="G. 파생/흐름")
fr = (useFunding and str.length(str.tostring(fundingSym))>0) ? request.security(fundingSym, timeframe.period, close, barmerge.gaps_off, barmerge.lookahead_off) : na
frOK_L = not useFunding or na(fr) or (fr <= maxLongFR/100.0)
frOK_S = not useFunding or na(fr) or (fr >= maxShortNeg/100.0)
useBasis     = input.bool(false,"베이시스(Perp-Spot) 필터", group="G. 파생/흐름")
spotSym      = input.symbol("BINANCE:SOLUSDT","스팟 심볼", group="G. 파생/흐름")
spotClose = request.security(spotSym, timeframe.period, close, lookahead=barmerge.lookahead_off)
basisPct  = (close/spotClose - 1.0) * 100.0
maxLongBasis = input.float(0.6,"롱 허용 최대 베이시스%", group="G. 파생/흐름")
minShortBasis= input.float(-0.6,"숏 허용 최소 베이시스%", group="G. 파생/흐름")
basisOK_L = not useBasis or (basisPct <= maxLongBasis)
basisOK_S = not useBasis or (basisPct >= minShortBasis)
useOI      = input.bool(false, "OI 증가 동행 요구", group="G. 파생/흐름")
oiSymbol   = input.symbol("", "OI 심볼", group="G. 파생/흐름")
oi = (useOI and str.length(str.tostring(oiSymbol))>0) ? request.security(oiSymbol, timeframe.period, close, lookahead=barmerge.lookahead_off) : na
deltaOI = na(oi) ? na : (oi - oi[1])
normDeltaOI = na(deltaOI) ? na : (deltaOI / math.max(1e-9, ta.sma(math.abs(deltaOI), 50)))
minDeltaOI = input.float(0.0, "정규화 ΔOI 최소치", group="G. 파생/흐름")
oiOK = not useOI or na(normDeltaOI) or (normDeltaOI >= minDeltaOI)
useDollarVol = input.bool(false, "유동성: 최소 달러거래대금 필터", group="G. 파생/흐름")
minDollarVol = input.float(1e6, "최소 달러 거래대금", group="G. 파생/흐름")
dollarVol = close * volume
liqOK = not useDollarVol or (dollarVol >= minDollarVol)

// ======================[ I. 엔트리 조건 (기존 모듈) ]=====================
rsi       = ta.rsi(close, rsiLen)
k         = ta.sma(ta.stoch(high, low, close, stochLen), 1)
dSig      = ta.sma(k, stochSig)
stochBull = ta.crossover(k, dSig) and k < 20
stochBear = ta.crossunder(k, dSig) and k > 80
touchLower = close <= bbDn or low <= bbDn
touchUpper = close >= bbUp or high >= bbUp
enableTrend   = input.bool(true, "트렌딩 모듈 ON", group="I. 엔트리")
enableRange   = input.bool(true, "레인지 모듈 ON",  group="I. 엔트리")
enableSqueeze = input.bool(true,"Quiet→Vol 돌파 모듈 ON", group="I. 엔트리")
useAVWAPfilter = input.bool(false, "AVWAP 방향 필터 사용", group="I. 엔트리")
avwapOK_L = not useAVWAPfilter or (useAnchors and not na(anchoredVWAP) and close >= anchoredVWAP)
avwapOK_S = not useAVWAPfilter or (useAnchors and not na(anchoredVWAP) and close <= anchoredVWAP)
trendLong  = enableTrend and (ta.adx(adxLen) >= adxTrend) and trendUp   and vwapOK_Long and inSess and htfOK and avwapOK_L
trendLong  := trendLong and (ta.crossover(close, emaFast) or stochBull or (useDonchian and ta.crossover(close, ta.highest(high, donLenEff))))
trendShort = enableTrend and (ta.adx(adxLen) >= adxTrend) and trendDown and vwapOK_Short and inSess and htfOK and avwapOK_S
trendShort := trendShort and (ta.crossunder(close, emaFast) or stochBear or (useDonchian and ta.crossunder(close, ta.lowest(low, donLenEff))))
rangeLong  = enableRange and (ta.adx(adxLen) <= adxRange) and inSess and htfOK and (rsi < rsiLow)  and (touchLower or ta.crossover(rsi, rsiLow))
rangeShort = enableRange and (ta.adx(adxLen) <= adxRange) and inSess and htfOK and (rsi > rsiHigh) and (touchUpper or ta.crossunder(rsi, rsiHigh))
squeezeLong  = enableSqueeze and vwapOK_Long  and inSess and htfOK and (not useQuiet2Vol or (isVolNow and isQuiet[1] and close > bbUp))
squeezeShort = enableSqueeze and vwapOK_Short and inSess and htfOK and (not useQuiet2Vol or (isVolNow and isQuiet[1] and close < bbDn))
orbOK = not useORB or isORBWindow

// ======================[ I2. Connors R3 모듈 ]=====================
groupR3 = "I2. Connors R3 (RSI 평균회귀)"
enableR3        = input.bool(true,  "R3 모듈 ON", group=groupR3)
r3_rsiLen       = input.int(2,      "RSI 기간", group=groupR3)
r3_consecDays   = input.int(3,      "연속 하락/상승 일수", minval=2, maxval=6, group=groupR3)
r3_useRSIstreak = input.bool(true,  "연속성 기준: RSI가 연속 하락/상승(OFF=종가)", group=groupR3)
r3_firstDayMaxRSI_Long  = input.float(60.0, "롱: 첫날 RSI 최대값(<)", step=0.5, group=groupR3)
r3_entryRSI_Long        = input.float(10.0, "롱 진입 RSI 임계(<)", step=0.5, group=groupR3)
r3_firstDayMinRSI_Short = input.float(60.0, "숏: 첫날 RSI 최소값(>)", step=0.5, group=groupR3)
r3_entryRSI_Short       = input.float(90.0, "숏 진입 RSI 임계(>)", step=0.5, group=groupR3)
r3_exitRSI_Long         = input.float(70.0, "롱 청산 RSI 임계(>)", step=0.5, group=groupR3)
r3_exitRSI_Short        = input.float(30.0, "숏 청산 RSI 임계(<)", step=0.5, group=groupR3)
r3_maTF         = input.timeframe("D", "장기 추세 MA TF", group=groupR3)
r3_maLen        = input.int(200, "장기 추세 MA 길이", group=groupR3)
r3_exactStreak  = input.bool(true, "연속 일수 정확히 일치만 허용(OFF=이상)", group=groupR3)
r3_timeStopBars = input.int(0,   "R3 전용 시간 스톱(바, 0=off)", group=groupR3)
r3_hardStopPct  = input.float(0.0,"R3 전용 하드 스톱%(0=off)", step=0.1, group=groupR3)
r3_exitMode     = input.string("Pro_Exit_Pack","R3 청산방식", options=["Classic_RSI","Pro_Exit_Pack"], group=groupR3)
r3_respectGates = input.bool(true, "R3도 글로벌 게이트(CS/RV/BTC/펀딩 등) 준수", group=groupR3)

rsi2 = ta.rsi(close, r3_rsiLen)
var int rsiDownCnt = 0, rsiUpCnt = 0, pxDownCnt = 0, pxUpCnt = 0
rsiDownCnt := (rsi2 < rsi2[1]) ? nz(rsiDownCnt[1]) + 1 : 0
rsiUpCnt   := (rsi2 > rsi2[1]) ? nz(rsiUpCnt[1]) + 1   : 0
pxDownCnt  := (close < close[1]) ? nz(pxDownCnt[1]) + 1 : 0
pxUpCnt    := (close > close[1]) ? nz(pxUpCnt[1]) + 1   : 0

streakDown = r3_useRSIstreak ? rsiDownCnt : pxDownCnt
streakUp   = r3_useRSIstreak ? rsiUpCnt   : pxUpCnt

r3_MA = request.security(syminfo.tickerid, r3_maTF, ta.sma(close, r3_maLen), lookahead=barmerge.lookahead_off)

firstRSI_Long_OK  = rsi2[r3_consecDays-1] < r3_firstDayMaxRSI_Long
firstRSI_Short_OK = rsi2[r3_consecDays-1] > r3_firstDayMinRSI_Short

streakOK_long  = r3_exactStreak ? (streakDown == r3_consecDays) : (streakDown >= r3_consecDays)
streakOK_short = r3_exactStreak ? (streakUp   == r3_consecDays) : (streakUp   >= r3_consecDays)

r3_long_base  = enableR3 and (close > r3_MA) and streakOK_long  and firstRSI_Long_OK  and (rsi2 <  r3_entryRSI_Long)
r3_short_base = enableR3 and (close < r3_MA) and streakOK_short and firstRSI_Short_OK and (rsi2 >  r3_entryRSI_Short)

// ======================[ J. CUSUM 이벤트 게이트 ]=====================
useCUSUM = input.bool(false, "CUSUM 변동 이벤트 게이트", group="J. CUSUM/게이트")
hCUSUM   = input.float(0.75, "CUSUM 임계값(표준화)", step=0.05, group="J. CUSUM/게이트")
retLog = math.log(close/close[1])
mu = ta.sma(retLog, 100)
sigma = ta.stdev(retLog, 100)
z = sigma>0? (retLog-mu)/sigma : 0.0
var float cpos=0, cneg=0
cpos := math.max(0, cpos + z - hCUSUM)
cneg := math.min(0, cneg + z + hCUSUM)
cusumEvent = (cpos==0 and z<0) or (cneg==0 and z>0)
cusumOK = not useCUSUM or cusumEvent

// ======================[ K. 게이트/진입 최종 ]=====================
dateOK  = inDate and (not (avoidWeekend and isWeekend)) and utcOK and embargoOK
flowOK_L= (btcOK_L and frOK_L and basisOK_L and oiOK and liqOK)
flowOK_S= (btcOK_S and frOK_S and basisOK_S and oiOK and liqOK)

enterLongBase0  = (trendLong or rangeLong or squeezeLong)
enterShortBase0 = (trendShort or rangeShort or squeezeShort)

r3GateOK_L = not r3_respectGates or (csOK_Long and rvOK and flowOK_L)
r3GateOK_S = not r3_respectGates or (csOK_Short and rvOK and flowOK_S)
enterLongBaseR3  = r3_long_base  and r3GateOK_L
enterShortBaseR3 = r3_short_base and r3GateOK_S

enterLongBase  = dateOK and orbOK and cusumOK and (enterLongBase0 or enterLongBaseR3)
enterShortBase = dateOK and orbOK and cusumOK and (enterShortBase0 or enterShortBaseR3)

enterLong  = barstate.isconfirmed and enterLongBase
enterShort = barstate.isconfirmed and enterShortBase

// ======================[ L. 리스크/사이징 + 적응 스로틀 ]=====================
groupRisk = "L. 리스크/사이징/스로틀"
riskPctPerTr = input.float(1.0, "트레이드당 위험% (자본 대비)", minval=0.05, maxval=10.0, group=groupRisk)
useVolSize   = input.bool(true,  "ATR-리스크 기반 사이징", group=groupRisk)
stopDistPctBase  = (ta.atr(atrLen) * atrTrailMult) / close * 100.0
minStopPct   = 0.01
basePct      = (useVolSize ? (riskPctPerTr / math.max(minStopPct, stopDistPctBase)) * 100.0 : default_qty_value)
basePct      := math.clamp(basePct, 0.1, 1000)
// Kelly
useKellyCap  = input.bool(true,  "Kelly 기반 포지션 상한", group=groupRisk)
kellyTradesN = input.int(40,     "Kelly 추정용 트레이드 수", minval=10, maxval=500, group=groupRisk)
kellyMult    = input.float(0.5,  "Kelly 배수(0.5=Half)", minval=0.1, maxval=1.0, group=groupRisk)
kellyMaxPct  = input.float(25,   "Kelly 최대 상한(% equity)", minval=1, maxval=100, group=groupRisk)
var float[] kellyRets = array.new_float()
var float   kEntry    = na
var int     kDir      = 0
openedNow  = (strategy.position_size != 0 and strategy.position_size[1] == 0)
closedNow  = (strategy.position_size == 0 and strategy.position_size[1] != 0)
if openedNow
    kEntry := strategy.position_avg_price
    kDir   := strategy.position_size > 0 ? 1 : -1
if closedNow and not na(kEntry) and kDir != 0
    retTrade = kDir == 1 ? (close / kEntry - 1.0) : (kEntry / close - 1.0)
    array.push(kellyRets, retTrade)
    if array.size(kellyRets) > kellyTradesN
        array.shift(kellyRets)
    kEntry := na, kDir := 0
winSum=0.0, winCnt=0, lossSum=0.0, lossCnt=0
for i=0 to array.size(kellyRets)-1
    v = array.get(kellyRets,i)
    if v>0
        winSum+=v
        winCnt+=1
    else if v<0
        lossSum+=-v
        lossCnt+=1
p = (winCnt+lossCnt)>0 ? winCnt/(winCnt+lossCnt) : na
Rwl = (winCnt>0 and lossCnt>0 and (lossSum/lossCnt)>0) ? ((winSum/winCnt)/(lossSum/lossCnt)) : na
fKelly = (not na(p) and not na(Rwl) and Rwl>0) ? (p - (1-p)/Rwl) : na
kellyCapPct = useKellyCap and not na(fKelly) ? math.clamp(100*kellyMult*fKelly, 0, kellyMaxPct) : na
useLossThrottle = input.bool(true, "손실연속 스로틀", group=groupRisk)
lossStepPct     = input.float(0.2, "손실 1회당 위험 감소율(예:0.2=20%)", minval=0.0, maxval=0.9, step=0.05, group=groupRisk)
maxThrottle     = input.float(0.6, "최대 위험 축소비율(예:0.6=60%)", minval=0.0, maxval=0.95, step=0.05, group=groupRisk)
var int lossStreak=0
if closedNow
    // 간단한 손익 추적: 마지막 거래 수익률 기준
    lastRet = array.size(kellyRets)>0 ? array.get(kellyRets, array.size(kellyRets)-1) : na
    lossStreak := (not na(lastRet) and lastRet<0) ? lossStreak+1 : 0
throttle = useLossThrottle ? math.min(maxThrottle, lossStreak*lossStepPct) : 0.0
basePct := basePct * (1.0 - throttle)
finalPct = (useKellyCap and not na(kellyCapPct)) ? math.min(basePct, kellyCapPct) : basePct

// ======================[ M. 터틀 피라미딩(선택) ]=====================
usePyramid   = input.bool(true, "터틀 피라미딩", group="M. 피라미딩/증액")
maxUnits     = input.int(4, "최대 유닛(≤4)", minval=1, maxval=4, group="M. 피라미딩/증액")
unitAdd_N    = input.float(0.5, "추가 유닛 간격(N)", step=0.1, group="M. 피라미딩/증액")
unitPct      = input.float(25, "유닛당 % of equity", minval=1, maxval=100, group="M. 피라미딩/증액")
Nlen_pyr     = input.int(20, "N(ATR) 길이", group="M. 피라미딩/증액")
Natr         = ta.atr(Nlen_pyr)
var int unitsNow=0
if openedNow
    unitsNow := 1
var float lastAddPx = na
if openedNow
    lastAddPx := strategy.position_avg_price
canAddLong  = usePyramid and strategy.position_size > 0 and unitsNow < maxUnits and close >= nz(lastAddPx) + unitAdd_N*Natr
canAddShort = usePyramid and strategy.position_size < 0 and unitsNow < maxUnits and close <= nz(lastAddPx) - unitAdd_N*Natr
if canAddLong
    strategy.entry("L.add."+str.tostring(unitsNow+1), strategy.long, qty_percent=unitPct)
    unitsNow += 1
    lastAddPx := close
if canAddShort
    strategy.entry("S.add."+str.tostring(unitsNow+1), strategy.short, qty_percent=unitPct)
    unitsNow += 1
    lastAddPx := close
if closedNow
    unitsNow := 0
    lastAddPx := na

// ======================[ N. 프로 EXIT PACK ]=====================
groupExit = "N. 프로 EXIT(실전)"
initStopMode = input.string("ATR_x", "초기 스톱 방식", options=["ATR_x","Turtle_2N","Chande-Kroll","Percent"], group=groupExit)
atrInitLen   = input.int(14, "ATR_x: ATR 길이", group=groupExit)
atrInitMult  = input.float(2.2, "ATR_x: 배수", step=0.1, group=groupExit)
N_len    = input.int(20, "Turtle N 길이(ATR)", group=groupExit)
N_mult   = input.float(2.0, "Turtle 초기스톱 배수(2N 권장)", step=0.1, group=groupExit)
ck_len1  = input.int(10, "Chande-Kroll L1", group=groupExit)
ck_len2  = input.int(20, "Chande-Kroll L2", group=groupExit)
ck_atr   = input.int(10, "Chande-Kroll ATR 길이", group=groupExit)
ck_mult  = input.float(1.5, "Chande-Kroll ATR 배수", step=0.1, group=groupExit)
pctStop  = input.float(1.5, "Percent: 초기 스톱% (예: 1.5=1.5%)", step=0.1, group=groupExit)
trail_Chandelier = input.bool(true,  "트레일링: Chandelier", group=groupExit)
chandLen         = input.int(22,     "Chandelier: ATR 길이", group=groupExit)
chandMult        = input.float(3.0,  "Chandelier: 배수", step=0.1, group=groupExit)
trail_Donchian   = input.bool(true,  "트레일링: Donchian", group=groupExit)
donTrailLen      = input.int(20,     "Donchian: 트레일 길이", group=groupExit)
trail_ATRx       = input.bool(true,  "트레일링: ATR x 배수", group=groupExit)
trail_ATRlen     = input.int(14,     "Trail ATR 길이", group=groupExit)
trail_ATRmult    = input.float(2.0,  "Trail ATR 배수", step=0.1, group=groupExit)
useBE_on_TP1     = input.bool(true,  "TP1 체결 시 BE(진입가로)", group=groupExit)
useBE_on_MFE     = input.bool(true,  "MFE가 R배수 이상이면 BE", group=groupExit)
be_MFE_R         = input.float(1.0,  "BE 트리거: MFE ≥ R", step=0.1, group=groupExit)
useTimeStop      = input.bool(true,  "시간 스톱", group=groupExit)
timeStopBars     = input.int(60,     "시간 스톱: 보유 바 수", minval=1, group=groupExit)
timeStopNeedProfit = input.bool(false, "시간 스톱은 미수익일 때만", group=groupExit)
useTPs      = input.bool(true, "TP1/TP2 분할청산", group=groupExit)
tp1_RR      = input.float(1.5, "TP1 RR", group=groupExit)
tp2_RR      = input.float(3.0, "TP2 RR", group=groupExit)
tp1_pct     = input.float(50, "TP1 % 청산", minval=1, maxval=99, group=groupExit)

// === 진입 ===
var int r3PosTag = 0  // 1=R3롱, -1=R3숏, 0=기타
r3EnterLongNow  = enableR3 and enterLongBaseR3 and (strategy.position_size <= 0)
r3EnterShortNow = enableR3 and enterShortBaseR3 and (strategy.position_size >= 0)

if enterLong and strategy.position_size <= 0
    strategy.entry("L", strategy.long,  qty_percent=finalPct)
    r3PosTag := r3EnterLongNow ? 1 : 0
if enterShort and strategy.position_size >= 0
    strategy.entry("S", strategy.short, qty_percent=finalPct)
    r3PosTag := r3EnterShortNow ? -1 : 0
if closedNow
    r3PosTag := 0

// === 초기 스톱 계산 ===
atrInit = ta.atr(atrInitLen)
N      = ta.atr(N_len)
ckATR  = ta.atr(ck_atr)
var float initStopL = na, initStopS = na, entryPx = na
openedNow2  = (strategy.position_size != 0 and strategy.position_size[1] == 0)
if openedNow2
    entryPx := strategy.position_avg_price
    if strategy.position_size > 0
        initStopL := switch initStopMode
            "ATR_x"        => entryPx - atrInitMult * atrInit
            "Turtle_2N"    => entryPx - N_mult * N
            "Chande-Kroll" => math.max(ta.lowest(low, ck_len2), ta.lowest(low, ck_len1)) - ck_mult * ckATR
            => entryPx * (1.0 - pctStop/100.0)
        initStopS := na
    else if strategy.position_size < 0
        initStopS := switch initStopMode
            "ATR_x"        => entryPx + atrInitMult * atrInit
            "Turtle_2N"    => entryPx + N_mult * N
            "Chande-Kroll" => math.min(ta.highest(high, ck_len2), ta.highest(high, ck_len1)) + ck_mult * ckATR
            => entryPx * (1.0 + pctStop/100.0)
        initStopL := na

// === MFE/BE 계산을 위한 R ===
var float riskPerTrade = na
if openedNow2
    riskPerTrade := strategy.position_size > 0 ? (entryPx - initStopL) : (initStopS - entryPx)
riskPerTrade := na(riskPerTrade) or riskPerTrade<=0 ? ta.atr(atrLen)*atrTrailMult : riskPerTrade

// Track HH/LL since entry for MFE and Chandelier
var float hhSinceEntry = na
var float llSinceEntry = na
if openedNow2
    hhSinceEntry := high
    llSinceEntry := low
else
    if strategy.position_size > 0
        hhSinceEntry := na(hhSinceEntry) ? high : math.max(hhSinceEntry, high)
    if strategy.position_size < 0
        llSinceEntry := na(llSinceEntry) ? low  : math.min(llSinceEntry, low)

longMFE  = strategy.position_size > 0 and not na(entryPx) and not na(hhSinceEntry) ? (hhSinceEntry - entryPx) : na
shortMFE = strategy.position_size < 0 and not na(entryPx) and not na(llSinceEntry) ? (entryPx - llSinceEntry) : na
mfeR = strategy.position_size > 0 ? (longMFE / math.max(1e-9, riskPerTrade)) :
       strategy.position_size < 0 ? (shortMFE/ math.max(1e-9, riskPerTrade)) : na

// === 트레일링 후보들 ===
chATR = ta.atr(chandLen)
chL = strategy.position_size > 0 and not na(hhSinceEntry) ? hhSinceEntry - chandMult * chATR : na
chS = strategy.position_size < 0 and not na(llSinceEntry) ? llSinceEntry + chandMult * chATR : na
donTrailL = strategy.position_size > 0 ? ta.lowest(low, donTrailLen) : na
donTrailS = strategy.position_size < 0 ? ta.highest(high, donTrailLen) : na
tATR = ta.atr(trail_ATRlen)
atrTrailL = strategy.position_size > 0 ? (close - trail_ATRmult * tATR) : na
atrTrailS = strategy.position_size < 0 ? (close + trail_ATRmult * tATR) : na

beTriggered = false
if strategy.position_size > 0
    beTriggered := (useBE_on_TP1 and close >= entryPx + riskPerTrade*tp1_RR) or (useBE_on_MFE and not na(mfeR) and mfeR >= be_MFE_R)
if strategy.position_size < 0
    beTriggered := (useBE_on_TP1 and close <= entryPx - riskPerTrade*tp1_RR) or (useBE_on_MFE and not na(mfeR) and mfeR >= be_MFE_R)
bePriceL = entryPx
bePriceS = entryPx

var float finalStopL = na
var float finalStopS = na

if strategy.position_size > 0
    float s1 = na(initStopL) ? -1e10 : initStopL
    float s2 = (trail_Chandelier and not na(chL)) ? chL : -1e10
    float s3 = (trail_Donchian   and not na(donTrailL)) ? donTrailL : -1e10
    float s4 = (trail_ATRx       and not na(atrTrailL)) ? atrTrailL : -1e10
    float s5 = beTriggered ? bePriceL : -1e10
    finalStopL := math.max(math.max(s1, s2), math.max(math.max(s3, s4), s5))
else
    finalStopL := na

if strategy.position_size < 0
    float t1 = na(initStopS) ? 1e10 : initStopS
    float t2 = (trail_Chandelier and not na(chS)) ? chS : 1e10
    float t3 = (trail_Donchian   and not na(donTrailS)) ? donTrailS : 1e10
    float t4 = (trail_ATRx       and not na(atrTrailS)) ? atrTrailS : 1e10
    float t5 = beTriggered ? bePriceS : 1e10
    finalStopS := math.min(math.min(t1, t2), math.min(math.min(t3, t4), t5))
else
    finalStopS := na

tp1L  = strategy.position_size > 0 ? entryPx + riskPerTrade * tp1_RR : na
tp2L  = strategy.position_size > 0 ? entryPx + riskPerTrade * tp2_RR : na
tp1S  = strategy.position_size < 0 ? entryPx - riskPerTrade * tp1_RR : na
tp2S  = strategy.position_size < 0 ? entryPx - riskPerTrade * tp2_RR : na

// 시간 스톱/보유 제한(공용)
var int barsInPos = 0
barsInPos := strategy.position_size != 0 ? nz(barsInPos) + 1 : 0
timeStopHit = false
if useTimeStop and strategy.position_size != 0
    unreal = strategy.position_size > 0 ? (close - entryPx) : (entryPx - close)
    timeStopHit := (barsInPos >= timeStopBars) and (not timeStopNeedProfit or unreal <= 0)

// === 주문/청산(공용) ===
if strategy.position_size > 0
    strategy.exit("XL_TP1", from_entry="L", stop = timeStopHit ? close : finalStopL, limit= useTPs ? tp1L : na, qty_percent= useTPs ? tp1_pct : 100)
    if useTPs
        strategy.exit("XL_TP2", from_entry="L", stop = timeStopHit ? close : finalStopL, limit=tp2L, qty_percent=100 - tp1_pct)
if strategy.position_size < 0
    strategy.exit("XS_TP1", from_entry="S", stop = timeStopHit ? close : finalStopS, limit= useTPs ? tp1S : na, qty_percent= useTPs ? tp1_pct : 100)
    if useTPs
        strategy.exit("XS_TP2", from_entry="S", stop = timeStopHit ? close : finalStopS, limit=tp2S, qty_percent=100 - tp1_pct)

// === R3 전용 청산 (Classic 모드 선택 시, R3 태그 포지션에만 적용) ===
r3ExitLong = enableR3 and (r3_exitMode=="Classic_RSI") and (r3PosTag==1) and (rsi2 > r3_exitRSI_Long)
r3ExitShort= enableR3 and (r3_exitMode=="Classic_RSI") and (r3PosTag==-1) and (rsi2 < r3_exitRSI_Short)
if r3ExitLong and strategy.position_size > 0
    strategy.close(id="L", comment="R3 RSI Exit L")
if r3ExitShort and strategy.position_size < 0
    strategy.close(id="S", comment="R3 RSI Exit S")

// === R3 전용 시간/하드 스톱(선택) ===
if enableR3 and r3PosTag!=0 and strategy.position_size!=0
    if r3_timeStopBars>0 and barsInPos>=r3_timeStopBars
        strategy.close_all(comment="R3 Time Stop")
    if r3_hardStopPct>0
        if r3PosTag==1 and (close <= entryPx*(1.0 - r3_hardStopPct/100.0))
            strategy.close_all(comment="R3 Hard Stop L")
        if r3PosTag==-1 and (close >= entryPx*(1.0 + r3_hardStopPct/100.0))
            strategy.close_all(comment="R3 Hard Stop S")

// 보유시간 하드 캡(옵션)
if (maxBarsHold > 0 and barsInPos >= maxBarsHold) and strategy.position_size != 0
    strategy.close_all(comment="Time Exit MaxHold")
    barsInPos := 0

// ======================[ O. 데일리 손실캡 & 드로우다운 킬스위치 ]=====================
groupDD = "O. 리스크 캡/쿨다운"
useDailyCap   = input.bool(true,  "일일 손실 캡/중지", group=groupDD)
capPct        = input.float(3.0,  "일손실 캡 % (자본 대비)", step=0.1, group=groupDD)
cooldownBars  = input.int(60,     "캡 히트 후 쿨다운 바", minval=0, group=groupDD)
var float dayStartEq = na
var int   cooldown=0
newDayTV = ta.change(time("D"))
if newDayTV
    dayStartEq := strategy.equity
    cooldown := 0
dayStartEq := na(dayStartEq) ? strategy.equity : dayStartEq
todayPnLPct = (strategy.equity - dayStartEq) / dayStartEq * 100.0
capHit = useDailyCap and todayPnLPct <= -capPct
cooldown := capHit ? cooldownBars : math.max(0, cooldown - 1)
useKillDD = input.bool(true, "MDD 킬스위치", group=groupDD)
killPct   = input.float(20.0, "최대낙폭 허용 %", step=0.5, group=groupDD)
var float eqHigh = na
eqHigh := na(eqHigh) ? strategy.equity : math.max(eqHigh, strategy.equity)
mddPct = (strategy.equity - eqHigh) / eqHigh * 100.0
killHit = useKillDD and (mddPct <= -killPct)

entryGateOK = not capHit and cooldown==0 and not killHit
enterLong  := enterLong  and entryGateOK
enterShort := enterShort and entryGateOK

// =========================[ P. 시각화 ]========================
plot(emaFast, "EMA Fast", color=color.new(color.teal, 0))
plot(emaSlow, "EMA Slow", color=color.new(color.orange, 0))
plot(bbUp,   "BB Upper",  color=color.new(color.gray, 50))
plot(bbDn,   "BB Lower",  color=color.new(color.gray, 50))
plot(stLine, "Supertrend", color=color.new(color.fuchsia, 0), display=(trendFilter=="Supertrend")? display.all : display.none)
plot(kama,   "KAMA",       color=color.new(color.yellow, 0),  display=(trendFilter=="KAMA")?       display.all : display.none)
plot(sessVWAP, "세션 VWAP", color=color.new(color.blue,60), display=useAVWAP?display.all:display.none)
plot(anchoredVWAP, "Anchored VWAP", color=color.new(color.aqua,0), display=useAnchors?display.all:display.none)
plot(r3_MA, "R3 200MA(TF)", color=color.new(color.purple, 0), display=enableR3?display.all:display.none)
bgcolor( (ta.adx(adxLen)>=adxTrend and trendUp) ? color.new(color.green,85) :
       (ta.adx(adxLen)<=adxRange) ? color.new(color.blue,85)  :
       isQuiet ? color.new(color.gray,88)  :
       isVolNow? color.new(color.red,88)   : na)

// ======================[ Q. 알림(상시) ]=====================
msgLong  = '{"side":"buy","symbol":"{{ticker}}","price":"{{close}}","id":"KASIA_RA_v5_2_R3","ts":"{{timenow}}"}'
msgShort = '{"side":"sell","symbol":"{{ticker}}","price":"{{close}}","id":"KASIA_RA_v5_2_R3","ts":"{{timenow}}"}'
alertcondition(enterLong,  title="Long Entry",  message=msgLong)
alertcondition(enterShort, title="Short Entry", message=msgShort)
alertcondition(strategy.position_size>0 and not na(finalStopL) and close <= finalStopL, title="Long Stop",  message='{"side":"close_long","symbol":"{{ticker}}","price":"{{close}}","id":"KASIA_RA_v5_2_R3"}')
alertcondition(strategy.position_size<0 and not na(finalStopS) and close >= finalStopS, title="Short Stop", message='{"side":"close_short","symbol":"{{ticker}}","price":"{{close}}","id":"KASIA_RA_v5_2_R3"}')
alertcondition(enableR3 and (r3ExitLong or r3ExitShort), title="R3 Classic Exit", message='{"side":"close","reason":"R3_RSI","symbol":"{{ticker}}","price":"{{close}}","id":"KASIA_RA_v5_2_R3"}')

7. 7번스크립트 A-ICT

//@version=5
strategy('A-ICT — Full Visuals Strategy NR (HTF Signal, Tick Entry, No-Repaint 2)', overlay=true, initial_capital=100000, commission_type=strategy.commission.percent, commission_value=0.05, calc_on_every_tick=true, pyramiding=0, default_qty_type=strategy.percent_of_equity, default_qty_value=1, max_boxes_count=500, max_lines_count=500, max_labels_count=500)

//==============================================================================
// 🎯 COMPREHENSIVE USER GUIDE & CONCEPTUAL FRAMEWORK
//==============================================================================
//
// 📊 ADVANCED ICT THEORY (AIT) - INSTITUTIONAL MARKET MANIPULATION DETECTOR
//
// This enhanced version includes:
// 1. Pattern identification boxes with proper classification
// 2. Delayed classification for accurate pattern detection
// 3. Full dashboard with all metrics
// 4. Separate key/guide dashboard
// 5. Complete visual system with lines, boxes, and labels
//
//==============================================================================
// 📊 Advanced ICT Theory INPUT CONFIGURATION
//==============================================================================
group_detection = "🎯 AIT Detection Engine"
minDisplacementCandles = input.int(2, 'Min Displacement Candles', minval=1, tooltip='🎯 WHAT IT IS: The minimum number of consecutive displacement candles required to validate an ICT element like OB or FVG.\n\n⚡ HOW IT WORKS: Displacement candles show strong momentum (large body, high volume). This setting ensures only significant moves form elements.\n\n📈 HIGHER VALUES (3-5): More selective, identifies major institutional moves. Fewer but higher-quality elements.\n📉 LOWER VALUES (1-2): More sensitive, captures minor displacements. Useful for scalping.\n\n🕒 TIMEFRAME OPTIMIZATION:\n• Scalping (1-5min): 1-2 (quick detections)\n• Day Trading (15min-1H): 2-3 (balanced)\n• Swing Trading (4H-1D): 3-5 (major moves)\n\n🏦 SECTOR RECOMMENDATIONS:\n• Forex: 2 (clean price action)\n• Crypto: 2-3 (volatility)\n• Stocks: 3 (institutional focus)\n\n💡 PRO TIP: Start with 2. Increase if too many weak elements appear.', group=group_detection)
mitigationMethod = input.string('Cross', 'Mitigation Method', options=['Cross', 'Close'], tooltip='🎯 WHAT IT IS: Defines how element mitigation (break) is confirmed in ICT structure.\n\n⚡ HOW IT WORKS:\n• Cross: Price crosses the level (including wicks) - faster but riskier.\n• Close: Candle must close beyond the level - more confirmation.\n\n📈 CROSS: Responsive for momentum trades.\n📉 CLOSE: Conservative, reduces false breaks.\n\n🕒 TIMEFRAME OPTIMIZATION:\n• Lower TFs: Cross (quick signals)\n• Higher TFs: Close (reliable breaks)\n\n💡 PRO TIP: Use Cross in trending markets, Close in ranging ones.', group=group_detection)
minElementSize = input.float(0.5, 'Min Element Size (ATR)', minval=0.1, step=0.1, tooltip='🎯 WHAT IT IS: Minimum size filter for elements, as ATR multiple.\n\n⚡ HOW IT WORKS: Elements smaller than ATR × value are ignored, filtering noise.\n\n📈 HIGHER VALUES (0.8-1.5): Only large, significant elements.\n📉 LOWER VALUES (0.1-0.4): Includes smaller elements.\n\n🕒 TIMEFRAME OPTIMIZATION:\n• Short-term: 0.3-0.5\n• Long-term: 0.7-1.0\n\n💡 PRO TIP: 0.5 balances sensitivity and quality.', group=group_detection)
maxHistoryBars = input.int(300, 'Max History Bars', minval=100, maxval=500, tooltip='🎯 WHAT IT IS: Limits historical bars analyzed for elements.\n\n⚡ HOW IT WORKS: Focuses on recent data for performance.\n\n📈 HIGHER VALUES (400-500): More history, better context.\n📉 LOWER VALUES (100-200): Faster, recent focus.\n\n💡 PRO TIP: 300 is optimal for most setups.', group=group_detection)
brokenAgeThreshold = input.int(50, 'Age Threshold', minval=20, tooltip='🎯 WHAT IT IS: Bars after which mitigated elements are removed.\n\n⚡ HOW IT WORKS: Clears old visuals to reduce clutter.\n\n📈 HIGHER VALUES (60-100): Keeps more history.\n📉 LOWER VALUES (20-40): Cleaner chart.\n\n💡 PRO TIP: 50 balances relevance and clarity.', group=group_detection)
pendingTimeout = input.int(10, 'Pending Timeout', minval=5, maxval=20, tooltip='🎯 WHAT IT IS: Bars after which unclassified pending elements are removed.\n\n⚡ HOW IT WORKS: Prevents stale pending items.\n\n📈 HIGHER VALUES (15-20): Gives more time for classification.\n📉 LOWER VALUES (5-10): Quicker cleanup.\n\n💡 PRO TIP: 10 works well for dynamic markets.', group=group_detection)
minQualityThreshold = input.int(30, 'Min Quality Threshold', minval=0, maxval=100, tooltip='🎯 WHAT IT IS: Minimum quality score for displaying elements.\n\n⚡ HOW IT WORKS: Filters low-quality detections.\n\n📈 HIGHER VALUES (50-80): Only premium setups.\n📉 LOWER VALUES (10-30): More elements shown.\n\n💡 PRO TIP: Start at 30, increase for selectivity.', group=group_detection)
group_structure = "🏗️ Market Structure"
internalPivotLookback = input.int(3, 'Internal Structure Lookback', minval=2, maxval=5, tooltip='🎯 WHAT IT IS: Bars to look back for internal structure pivots.\n\n⚡ HOW IT WORKS: Detects minor swings for BOS/CHoCH.\n\n📈 HIGHER VALUES (4-5): Smoother internal structure.\n📉 LOWER VALUES (2-3): More sensitive to changes.\n\n💡 PRO TIP: 3 is standard for most TFs.', group=group_structure)
externalPivotLookback = input.int(10, 'External Structure Lookback', minval=5, maxval=20, tooltip='🎯 WHAT IT IS: Bars to look back for external structure pivots.\n\n⚡ HOW IT WORKS: Identifies major swings for overall trend.\n\n📈 HIGHER VALUES (15-20): Broader market context.\n📉 LOWER VALUES (5-10): Focus on recent structure.\n\n💡 PRO TIP: 10 balances detail and overview.', group=group_structure)
majorSwingLookback = input.int(20, 'Major Swing Lookback', minval=10, maxval=50, tooltip='🎯 WHAT IT IS: Bars to look back for major swing points.\n\n⚡ HOW IT WORKS: Detects high-level highs/lows for long-term bias.\n\n📈 HIGHER VALUES (30-50): Long-term swings.\n📉 LOWER VALUES (10-20): Medium-term focus.\n\n💡 PRO TIP: 20 for swing trading.', group=group_structure)
requireVolumeConfirm = input.bool(true, 'Require Volume Confirmation', tooltip='🎯 WHAT IT IS: Mandates above-average volume for structure breaks.\n\n⚡ HOW IT WORKS: Confirms BOS/CHoCH with volume spike.\n\n💡 PRO TIP: Enable for higher conviction, disable in low-volume markets.', group=group_structure)
requireCandleConfirm = input.bool(true, 'Require Candle Pattern Confirmation', tooltip='🎯 WHAT IT IS: Requires specific candle patterns for structure validation.\n\n⚡ HOW IT WORKS: Ensures breaks align with bullish/bearish candles.\n\n💡 PRO TIP: Enable to filter noise, disable for pure price action.', group=group_structure)
group_display = "📊 Display Configuration"
maxElementsToDisplay = input.int(10, 'Max Active Elements', minval=5, maxval=20, tooltip='🎯 WHAT IT IS: Maximum number of elements shown on chart.\n\n⚡ HOW IT WORKS: Limits visuals to recent/high-quality ones.\n\n📈 HIGHER VALUES: More elements visible.\n📉 LOWER VALUES: Cleaner chart.\n\n💡 PRO TIP: 10 prevents clutter.', group=group_display)
showIdentificationBoxes = input.bool(true, 'Show Identification Boxes', tooltip='🎯 WHAT IT IS: Displays boxes around detected patterns.\n\n⚡ HOW IT WORKS: Highlights OBs, FVGs, etc., for easy spotting.\n\n💡 PRO TIP: Essential for beginners.', group=group_display)
showPatternLines = input.bool(true, 'Show Pattern Lines', tooltip='🎯 WHAT IT IS: Draws lines for BOS/CHoCH patterns.\n\n⚡ HOW IT WORKS: Connects structure breaks visually.\n\n💡 PRO TIP: Keep on for trend analysis.', group=group_display)
showOriginLine = input.bool(true, 'Show Origin Level', tooltip='🎯 WHAT IT IS: Highlights the primary bias level.\n\n⚡ HOW IT WORKS: Shows origin of current structure.\n\n💡 PRO TIP: Key for directional bias.', group=group_display)
showMitigationLevels = input.bool(true, 'Show Mitigation Levels', tooltip='🎯 WHAT IT IS: Displays key mitigation zones.\n\n⚡ HOW IT WORKS: Marks broken levels as S/R.\n\n💡 PRO TIP: Use for targets/stops.', group=group_display)
showElementLines = input.bool(true, 'Show Element Lines', tooltip='🎯 WHAT IT IS: Draws core lines for elements.\n\n⚡ HOW IT WORKS: Visualizes midpoints of patterns.\n\n💡 PRO TIP: Fundamental feature.', group=group_display)
showTradeScore = input.bool(true, 'Show Trade Quality Score', tooltip='🎯 WHAT IT IS: Displays quality grades for setups.\n\n⚡ HOW IT WORKS: Grades [A+ to D] based on confluence.\n\n💡 PRO TIP: Focus on A/B grades.', group=group_display)
group_zones = "🔲 Identification Boxes Style"
boxStyle = input.string('Solid', 'Box Style', options=['Solid', 'Dashed', 'Dotted'], tooltip='🎯 WHAT IT IS: Appearance of pattern boxes.\n\n⚡ HOW IT WORKS: Solid for filled, Dashed/Dotted for outlines.\n\n💡 PRO TIP: Solid for emphasis.', group=group_zones)
type1BoxColor = input.color(#2962FF, 'Order Block Color', tooltip='🎯 WHAT IT IS: Color for Order Block boxes.\n\n💡 PRO TIP: Blue for bullish OBs.', group=group_zones)
type2BoxColor = input.color(#FF6B00, 'Trap Zone Color', tooltip='🎯 WHAT IT IS: Color for Trap Zone boxes.\n\n💡 PRO TIP: Orange for liquidity traps.', group=group_zones)
type3BoxColor = input.color(#00E676, 'Reversal Zone Color', tooltip='🎯 WHAT IT IS: Color for Reversal/S&R zones.\n\n💡 PRO TIP: Green for support areas.', group=group_zones)
fvgBoxColor = input.color(#9C27B0, 'FVG Color', tooltip='🎯 WHAT IT IS: Color for Fair Value Gap boxes.\n\n💡 PRO TIP: Purple for imbalance gaps.', group=group_zones)
boxTransparency = input.int(85, 'Box Transparency', minval=50, tooltip='🎯 WHAT IT IS: Opacity of boxes.\n\n📈 HIGHER: More transparent.\n📉 LOWER: More solid.\n\n💡 PRO TIP: 85 for subtle overlay.', group=group_zones)
group_levels = "📏 Level Appearance"
originLineWidth = input.int(3, 'Origin Width', minval=1, tooltip='🎯 WHAT IT IS: Thickness of origin lines.\n\n📈 HIGHER: Bolder lines.\n\n💡 PRO TIP: 3 for visibility.', group=group_levels)
mitigationLineWidth = input.int(2, 'Mitigation Width', minval=1, tooltip='🎯 WHAT IT IS: Thickness of mitigation lines.\n\n📈 HIGHER: Bolder.\n\n💡 PRO TIP: 2 for clarity.', group=group_levels)
patternLineWidth = input.int(2, 'Pattern Line Width', minval=1, tooltip='🎯 WHAT IT IS: Thickness of pattern lines.\n\n📈 HIGHER: Bolder.\n\n💡 PRO TIP: 2 standard.', group=group_levels)
labelPosition = input.int(5, 'Label Offset', minval=1, tooltip='🎯 WHAT IT IS: Distance of labels from bars.\n\n📈 HIGHER: Further right.\n\n💡 PRO TIP: 5 avoids overlap.', group=group_levels)
group_colors = "🎨 Color Scheme"
colorScheme = input.string('Professional', 'Theme', options=['Professional', 'Vibrant', 'Dark', 'Custom'], tooltip='🎯 WHAT IT IS: Preset themes for visuals.\n\n⚡ OPTIONS: Professional (clean), Vibrant (bold), Dark (subtle), Custom (user-defined).\n\n💡 PRO TIP: Professional for most users.', group=group_colors)
customOriginColor = input.color(#FFD700, 'Custom Origin', tooltip='🎯 WHAT IT IS: Custom color for origin levels in Custom theme.\n\n💡 PRO TIP: Gold for premium look.', group=group_colors)
customMitigationColor = input.color(#00BCD4, 'Custom Mitigation', tooltip='🎯 WHAT IT IS: Custom color for mitigation levels.\n\n💡 PRO TIP: Cyan for visibility.', group=group_colors)
g_dashboard = "📊 Main Dashboard Settings"
showDashboard = input.bool(true, "Show Main Dashboard", tooltip='🎯 WHAT IT IS: Toggle for main metrics dashboard.\n\n⚡ HOW IT WORKS: Displays live ICT stats.\n\n💡 PRO TIP: Always enable.', group = g_dashboard)
dashboardSize = input.string("Large", "Size", options = ["Small", "Medium", "Large"], tooltip='🎯 WHAT IT IS: Controls dashboard detail level.\n\n⚡ OPTIONS: Small (basic), Medium (expanded), Large (full with guide).\n\n💡 PRO TIP: Large for complete view.', group = g_dashboard)
dashboardPosition = input.string("Bottom Right", "Position", options = ["Top Right", "Top Left", "Bottom Right", "Bottom Left"], tooltip='🎯 WHAT IT IS: Screen position for dashboard.\n\n💡 PRO TIP: Bottom Right keeps price visible.', group = g_dashboard)
dashboardTransparency = input.int(20, "Transparency", minval = 0, tooltip='🎯 WHAT IT IS: Dashboard opacity.\n\n📈 HIGHER: More transparent.\n\n💡 PRO TIP: 20 for subtle overlay.', group = g_dashboard)
g_narrative = "📖 Narrative Dashboard Settings"
showNarrativeDashboard = input.bool(true, "Show Narrative Dashboard", tooltip='🎯 WHAT IT IS: Toggle for narrative analysis dashboard.\n\n⚡ HOW IT WORKS: Shows element details and market story.\n\n💡 PRO TIP: Enable for insights.', group = g_narrative)
narrativeDashboardPosition = input.string("Bottom Left", "Position", options = ["Top Right", "Top Left", "Bottom Right", "Bottom Left"], tooltip='🎯 WHAT IT IS: Screen position for narrative dashboard.\n\n💡 PRO TIP: Bottom Left for balance.', group = g_narrative)
narrativeTransparency = input.int(20, "Transparency", minval = 0, tooltip='🎯 WHAT IT IS: Narrative dashboard opacity.\n\n📈 HIGHER: More transparent.\n\n💡 PRO TIP: 20 for readability.', group = g_narrative)
group_mtf = "🕒 Multi-Timeframe Analysis"
showHTF = input.bool(true, "Show HTF Analysis", tooltip='🎯 WHAT IT IS: Enables higher timeframe alignment checks.\n\n⚡ HOW IT WORKS: Aligns signals with HTF bias.\n\n💡 PRO TIP: Essential for confluence.', group=group_mtf)
htfTimeframe = input.string("60", "HTF Timeframe", options=["15", "30", "60", "240", "D"], tooltip='🎯 WHAT IT IS: Higher timeframe for analysis.\n\n⚡ OPTIONS: 15min to Daily.\n\n💡 PRO TIP: Use 60min for intraday.', group=group_mtf)
htfAlignmentRequired = input.bool(true, "Require HTF Alignment", tooltip='🎯 WHAT IT IS: Mandates HTF agreement for validity.\n\n⚡ HOW IT WORKS: Filters misaligned signals.\n\n💡 PRO TIP: Enable for better accuracy.', group=group_mtf)
group_liquidity = "💧 Liquidity Mapping"
showLiquidity = input.bool(true, "Show Liquidity Levels", tooltip='🎯 WHAT IT IS: Displays buy/sell side liquidity lines.\n\n⚡ HOW IT WORKS: Tracks highs/lows for sweeps.\n\n💡 PRO TIP: Key for trap identification.', group=group_liquidity)
liquidityLookback = input.int(20, "Liquidity Lookback", minval=10, tooltip='🎯 WHAT IT IS: Bars back for liquidity calculation.\n\n📈 HIGHER: Broader liquidity view.\n\n💡 PRO TIP: 20 standard.', group=group_liquidity)
showSweptLiquidity = input.bool(true, "Show Swept Liquidity", tooltip='🎯 WHAT IT IS: Visualizes swept (broken) liquidity.\n\n⚡ HOW IT WORKS: Dotted lines for swept levels.\n\n💡 PRO TIP: Enable to track manipulations.', group=group_liquidity)
group_premium = "⚖️ Premium/Discount"
showPremiumDiscount = input.bool(true, "Show Premium/Discount Zones", tooltip='🎯 WHAT IT IS: Highlights premium/discount areas.\n\n⚡ HOW IT WORKS: Based on equilibrium calculation.\n\n💡 PRO TIP: Trade discounts in uptrends.', group=group_premium)
pdLookback = input.int(50, "P/D Lookback", minval=20, tooltip='🎯 WHAT IT IS: Bars for premium/discount calculation.\n\n📈 HIGHER: Smoother zones.\n\n💡 PRO TIP: 50 for accuracy.', group=group_premium)
group_sessions = "⏰ Sessions & Killzones"
showSessions = input.bool(true, "Show Trading Sessions", tooltip='🎯 WHAT IT IS: Colors session backgrounds.\n\n⚡ HOW IT WORKS: Asian (blue), London (yellow), NY (green).\n\n💡 PRO TIP: For session-based trading.', group=group_sessions)
showKillzones = input.bool(true, "Show ICT Killzones", tooltip='🎯 WHAT IT IS: Highlights high-volatility killzones.\n\n⚡ HOW IT WORKS: Red boxes for manipulation periods.\n\n💡 PRO TIP: Trade during killzones.', group=group_sessions)
group_ipda = "⏰ Interbank Price Delivery Algorithm - IPDA Windows"
showIPDA = input.bool(true, "Show IPDA Windows", tooltip='🎯 WHAT IT IS: Displays IPDA delivery periods.\n\n⚡ HOW IT WORKS: Blue (Asian), Orange (London Open), Purple (NY Open).\n\n💡 PRO TIP: Watch for deliveries in these windows.', group=group_ipda)
// ========================================
 // TYPE DEFINITIONS
// ========================================
type ICTElement
    float high = na
    float low = na
    int startBar = na
    int endBar = na
    bool mitigated = false
    int mitigationBar = na
    string elementType = "Pending"
    string timeframe = ""
    bool isOrigin = false
    bool isMitigation = false
    float strength = 0.5
    bool insideKillzone = false
    bool nearLiquidity = false
    float qualityScore = 0.0
    bool causedBOS = false
    bool inLogicalArea = false
    bool hasHighVolumeFVG = false
    string narrativeRole = ""
    box identificationBox = na
    line patternLine = na
    label patternLabel = na
type MarketStructure
    bool bullishBOS = false
    bool bearishBOS = false
    bool choch = false
    int lastBOSBar = na
    float lastBOSPrice = na
    bool internalBullishBOS = false
    bool internalBearishBOS = false
    bool externalBullishBOS = false
    bool externalBearishBOS = false
    float lastInternalHigh = na
    float lastInternalLow = na
    float lastExternalHigh = na
    float lastExternalLow = na
    string primaryTrend = "NEUTRAL"
    string orderFlow = "NEUTRAL"
type FVGPattern
    bool detected = false
    bool isBullish = false
    float high = na
    float low = na
    int startBar = na
    box fvgBox = na
    line fvgLine = na
    label fvgLabel = na
type ICTPattern
    string name
    bool detected = false
    int detectionBar = na
    float strength = 0.0
type TradeScore
    float momentum = 0
    float structure = 0
    float liquidity = 0
    float confluence = 0
    float total = 0
    string grade = ""
type TradeSuggestion
    string action = ""
    float entryPrice = 0
    float stopLoss = 0
    float takeProfit1 = 0
    float takeProfit2 = 0
    float riskReward = 0
type SmartMoneyFlow
    float buyVolume = 0
    float sellVolume = 0
    float netFlow = 0
    string bias = "NEUTRAL"
type MarketNarrative
    string currentPhase = ""
    string lastEvent = "No recent structural change"
    string nextExpectation = ""
    string causalSequence = ""
    int sequenceStartBar = 0
    float keyLevel = na
    string keySetup = ""
type IPDAWindow
    string name
    string startTime
    string endTime
    bool active
// ========================================
 // GLOBAL CALCULATIONS
// ========================================
volumeSMA20 = ta.sma(volume, 20)
volumeSMA10 = ta.sma(volume, 10)
atr14 = ta.atr(14)
rsi14 = ta.rsi(close, 14)
avgRange20 = ta.sma(high - low, 20)
trendSMA = ta.sma(close, 50)
internalHigh = ta.pivothigh(high, internalPivotLookback, internalPivotLookback)
internalLow = ta.pivotlow(low, internalPivotLookback, internalPivotLookback)
externalHigh = ta.pivothigh(high, externalPivotLookback, externalPivotLookback)
externalLow = ta.pivotlow(low, externalPivotLookback, externalPivotLookback)
majorHigh = ta.pivothigh(high, majorSwingLookback, majorSwingLookback)
majorLow = ta.pivotlow(low, majorSwingLookback, majorSwingLookback)
[htfH, htfL, htfC, htfO, htfV] = request.security(syminfo.tickerid, htfTimeframe, [high, low, close, open, volume], lookahead=barmerge.lookahead_off)
htfSMA20 = request.security(syminfo.tickerid, htfTimeframe, ta.sma(close, 20), lookahead=barmerge.lookahead_off)
htfTrend = htfC > htfSMA20 ? "BULLISH" : htfC < htfSMA20 ? "BEARISH" : "NEUTRAL"
htfBias = htfC > htfH[1] ? "BULLISH" : htfC < htfL[1] ? "BEARISH" : "NEUTRAL"
bsl = ta.highest(high, liquidityLookback)
ssl = ta.lowest(low, liquidityLookback)
bslSwept = high > bsl[1] and close < bsl[1]
sslSwept = low < ssl[1] and close > ssl[1]
pdHigh = ta.highest(high, pdLookback)
pdLow = ta.lowest(low, pdLookback)
equilibrium = (pdHigh + pdLow) / 2
isPremium = close > equilibrium + (pdHigh - equilibrium) * 0.5
isDiscount = close < equilibrium - (equilibrium - pdLow) * 0.5
isEquilibrium = not isPremium and not isDiscount
isAsianSession = not na(time(timeframe.period, "0000-0800", "UTC"))
isLondonSession = not na(time(timeframe.period, "0800-1200", "UTC"))
isNYSession = not na(time(timeframe.period, "1200-1600", "UTC"))
isLondonKZ = not na(time(timeframe.period, "0800-0900", "UTC"))
isNYAMKZ = not na(time(timeframe.period, "1330-1430", "UTC"))
isNYPMKZ = not na(time(timeframe.period, "1500-1600", "UTC"))
inKillzone = isLondonKZ or isNYAMKZ or isNYPMKZ
ipdaAsianRange = not na(time(timeframe.period, "0000-0300", "UTC"))
ipdaLondonOpen = not na(time(timeframe.period, "0800-0900", "UTC"))
ipdaNYOpen = not na(time(timeframe.period, "1330-1430", "UTC"))
// ========================================
 // COLOR THEMES
// ========================================
getColorTheme(theme) =>
    switch theme
        'Vibrant' => [color.new(#FFD700, 0), color.new(#00E5FF, 0), color.new(#FF4081, 0), color.new(#76FF03, 0)]
        'Dark' => [color.new(#FFA726, 0), color.new(#42A5F5, 0), color.new(#AB47BC, 0), color.new(#66BB6A, 0)]
        'Custom' => [customOriginColor, customMitigationColor, color.new(#808080, 0), color.new(#404040, 0)]
        => [color.new(#FFB800, 0), color.new(#00ACC1, 0), color.new(#7B1FA2, 0), color.new(#43A047, 0)]
[originColor, mitigationColor, brokenColor, bgColor] = getColorTheme(colorScheme)
// ========================================
 // GLOBAL VARIABLES
// ========================================
var array<ICTElement> elements = array.new<ICTElement>()
var array<line> elementLines = array.new<line>()
var array<label> elementLabels = array.new<label>()
var array<line> structureLines = array.new<line>()
var array<label> structureLabels = array.new<label>()
var array<FVGPattern> fvgPatterns = array.new<FVGPattern>()
var table dashboard = na
var table narrativeDashboard = na
var ICTElement currentOrigin = na
var array<ICTElement> mitigationLevels = array.new<ICTElement>()
var int lastDetectionBar = 0
var MarketStructure ms = MarketStructure.new()
var MarketNarrative narrative = MarketNarrative.new()
var box asianBox = na
var box londonBox = na
var box nyBox = na
var box londonKZBox = na
var box nyamKZBox = na
var box nypmKZBox = na
var box premiumBox = na
var box discountBox = na
var box equilibriumBox = na
// ========================================
 // CORE FUNCTIONS
// ========================================
safeGet(arr, index) =>
    size = array.size(arr)
    size > 0 and index >= 0 and index < size ? array.get(arr, index) : na
detectAdvancedStructure() =>
    ms.primaryTrend := close > trendSMA ? "BULLISH" : close < trendSMA ? "BEARISH" : "NEUTRAL"
    if not na(internalHigh)
        ms.lastInternalHigh := internalHigh
    if not na(internalLow)
        ms.lastInternalLow := internalLow
    if not na(externalHigh)
        ms.lastExternalHigh := externalHigh
    if not na(externalLow)
        ms.lastExternalLow := externalLow
    ms.internalBullishBOS := not na(ms.lastInternalHigh) and high > ms.lastInternalHigh
    ms.internalBearishBOS := not na(ms.lastInternalLow) and low < ms.lastInternalLow
    volumeConfirm = not requireVolumeConfirm or volume > volumeSMA20 * 1.2
    bullishCandleConfirm = not requireCandleConfirm or (close > open and close[1] > open[1])
    bearishCandleConfirm = not requireCandleConfirm or (close < open and close[1] < open[1])
    if not na(ms.lastExternalHigh) and high > ms.lastExternalHigh and volumeConfirm and bullishCandleConfirm
        ms.externalBullishBOS := true
        ms.bullishBOS := true
        ms.bearishBOS := false
        ms.lastBOSBar := bar_index
        ms.lastBOSPrice := high
        ms.orderFlow := "BULLISH"
    else
        ms.externalBullishBOS := false
    if not na(ms.lastExternalLow) and low < ms.lastExternalLow and volumeConfirm and bearishCandleConfirm
        ms.externalBearishBOS := true
        ms.bearishBOS := true
        ms.bullishBOS := false
        ms.lastBOSBar := bar_index
        ms.lastBOSPrice := low
        ms.orderFlow := "BEARISH"
    else
        ms.externalBearishBOS := false
    ms.choch := false
    if ms.orderFlow == "BULLISH" and ms.externalBearishBOS
        ms.choch := true
        narrative.lastEvent := "CHoCH: Bullish to Bearish"  
    if ms.orderFlow == "BEARISH" and ms.externalBullishBOS
        ms.choch := true
        narrative.lastEvent := "CHoCH: Bearish to Bullish"
detectFVG() =>
    fvg = FVGPattern.new()
    minGapSize = atr14 * 0.5
    bullishGap = low > high[2]
    if bullishGap
        gapSize = low - high[2]
        if gapSize > minGapSize and close > open and volume > volumeSMA20 * 1.2
            fvg.detected := true
            fvg.isBullish := true
            fvg.high := low
            fvg.low := high[2]
            fvg.startBar := bar_index - 2
    bearishGap = high < low[2]
    if bearishGap
        gapSize = low[2] - high
        if gapSize > minGapSize and close < open and volume > volumeSMA20 * 1.2
            fvg.detected := true
            fvg.isBullish := false
            fvg.high := low[2]
            fvg.low := high
            fvg.startBar := bar_index - 2       
    fvg
calculateTradeScore(element) =>
    score = TradeScore.new()
    if na(element)
        score
    else
        score.momentum := element.strength * 25
        if element.causedBOS
            score.structure := 25
        else if ms.externalBullishBOS and element.elementType == "Order Block"
            score.structure := 20
        else if element.elementType == "Trap Zone" and element.nearLiquidity
            score.structure := 15
        else if element.inLogicalArea
            score.structure := 10
        else
            score.structure := 5
        if element.nearLiquidity and (bslSwept or sslSwept)
            score.liquidity := 25
        else if element.nearLiquidity
            score.liquidity := 15
        else
            score.liquidity := 5
        confluenceFactors = 0
        if element.insideKillzone
            confluenceFactors += 2
        if element.inLogicalArea
            confluenceFactors += 2
        if htfBias == ms.orderFlow
            confluenceFactors += 3
        score.confluence := math.min(confluenceFactors * 3.5, 25)
        score.total := score.momentum + score.structure + score.liquidity + score.confluence
        score.grade := score.total >= 80 ? "A+" : score.total >= 65 ? "A" : score.total >= 50 ? "B" : score.total >= 40 ? "C" : "D"   
        score
calculateSmartMoneyFlow() =>
    flow = SmartMoneyFlow.new()
    flow.buyVolume := 0.0
    flow.sellVolume := 0.0
    flow.netFlow := 0.0
    flow.bias := "NEUTRAL"
    if volume > 0
        priceRange = math.max(high - low, 0.00001)
        if close > open
            flow.buyVolume := volume * (close - low) / priceRange
            flow.sellVolume := volume - flow.buyVolume
        else
            flow.sellVolume := volume * (high - close) / priceRange
            flow.buyVolume := volume - flow.sellVolume
        flow.netFlow := flow.buyVolume - flow.sellVolume
        if flow.netFlow > volumeSMA20 * 0.2
            flow.bias := "STRONG BULLISH FLOW"
        else if flow.netFlow > 0
            flow.bias := "BULLISH FLOW"
        else if flow.netFlow < -volumeSMA20 * 0.2
            flow.bias := "STRONG BEARISH FLOW"
        else if flow.netFlow < 0
            flow.bias := "BEARISH FLOW"
        else
            flow.bias := "NEUTRAL"
    flow
generateNarrativeTradeSuggestion(element) =>
    suggestion = TradeSuggestion.new()
    
    if na(element) or element.mitigated
        suggestion
    else
        score = calculateTradeScore(element)
        
        if score.total >= 55 and element.narrativeRole != ""
            if ms.orderFlow == "BULLISH" and element.elementType == "Type 1" and isDiscount
                suggestion.action := "LONG - " + element.narrativeRole
                suggestion.entryPrice := close
                suggestion.stopLoss := element.low - atr14 * 0.2
                suggestion.takeProfit1 := close + (close - suggestion.stopLoss) * 1.5
                suggestion.takeProfit2 := close + (close - suggestion.stopLoss) * 3
                if element.nearLiquidity
                    suggestion.takeProfit1 := bsl
            else if ms.orderFlow == "BEARISH" and element.elementType == "Type 1" and isPremium
                suggestion.action := "SHORT - " + element.narrativeRole
                suggestion.entryPrice := close
                suggestion.stopLoss := element.high + atr14 * 0.2
                suggestion.takeProfit1 := close - (suggestion.stopLoss - close) * 1.5
                suggestion.takeProfit2 := close - (suggestion.stopLoss - close) * 3
                if element.nearLiquidity
                    suggestion.takeProfit1 := ssl      
        if suggestion.action != ""
            suggestion.riskReward := math.abs(suggestion.takeProfit1 - suggestion.entryPrice) / math.abs(suggestion.entryPrice - suggestion.stopLoss)
        suggestion
detectAdvancedICTPatterns() =>
    patterns = array.new<ICTPattern>()
    breakerBlock = ICTPattern.new()
    breakerBlock.name := "Breaker Block"
    breakerBlock.detected := false
    if array.size(elements) >= 2
        lastElement = safeGet(elements, array.size(elements) - 1)
        prevElement = safeGet(elements, array.size(elements) - 2)
        if not na(lastElement) and not na(prevElement)
            if lastElement.mitigated and prevElement.mitigated
                if lastElement.elementType == "Type 1" and prevElement.elementType == "Type 2"
                    breakerBlock.detected := true
                    breakerBlock.detectionBar := bar_index
                    breakerBlock.strength := (lastElement.strength + prevElement.strength) / 2
                    narrative.lastEvent := "Breaker Block formed"
    array.push(patterns, breakerBlock)
    mitigationBlock = ICTPattern.new()
    mitigationBlock.name := "Mitigation Block"
    mitigationBlock.detected := false  
    if array.size(elements) >= 1
        lastElement = safeGet(elements, array.size(elements) - 1)
        if not na(lastElement) and lastElement.mitigated and lastElement.causedBOS
            mitigationBlock.detected := true
            mitigationBlock.detectionBar := bar_index
            mitigationBlock.strength := lastElement.strength           
    array.push(patterns, mitigationBlock)    
    patterns
buildMarketNarrative() =>
    sessionDetail = ""
    if inKillzone
        if isLondonKZ
            sessionDetail := "London KZ - Expect directional move"
        else if isNYAMKZ
            sessionDetail := "NY AM KZ - Peak volatility window"
        else if isNYPMKZ
            sessionDetail := "NY PM KZ - Position squaring"
        narrative.currentPhase := sessionDetail
    else if isAsianSession
        narrative.currentPhase := "Asian Range - Accumulation phase"
    else if isLondonSession
        narrative.currentPhase := "London Session - Expansion expected"
    else if isNYSession
        narrative.currentPhase := "NY Session - Continuation/Reversal"
    else
        narrative.currentPhase := "Inter-session - Low probability"
    sequence = ""
    recentLiquiditySweep = (bslSwept or sslSwept)
    recentBOS = bar_index - ms.lastBOSBar < 10
    hasUnmitigatedOB = false
    nearestOBLevel = 0.0
    strongestElement = ICTElement.new()
    if array.size(elements) > 0
        for i = array.size(elements) - 1 to math.max(0, array.size(elements) - 10)
            e = safeGet(elements, i)
            if not na(e) and not e.mitigated and e.elementType != "Pending"
                if e.qualityScore > 60
                    hasUnmitigatedOB := true
                    nearestOBLevel := (e.high + e.low) / 2
                    if e.qualityScore > strongestElement.qualityScore
                        strongestElement := e
    if recentLiquiditySweep
        sequence += "💧 " + (bslSwept ? "Buy-side" : "Sell-side") + " liquidity swept → "
        narrative.keyLevel := bslSwept ? bsl : ssl
    if ms.choch
        sequence += "⚡ CHoCH confirmed → Bias shift to " + ms.orderFlow + " → "
    if recentBOS
        bosType = ms.externalBullishBOS ? "Bullish" : ms.externalBearishBOS ? "Bearish" : "Internal"
        sequence += "📊 " + bosType + " BOS → "
    if hasUnmitigatedOB
        sequence += "🎯 Active " + strongestElement.elementType + " at " + str.tostring(nearestOBLevel, "#.####") + " → "
    expectation = ""
    if recentLiquiditySweep and not recentBOS
        if bslSwept
            expectation := "🔻 Reversal likely after BSL sweep. Watch for bearish OB formation near " + str.tostring(bsl, "#.####")
        else
            expectation := "🔺 Reversal likely after SSL sweep. Watch for bullish OB formation near " + str.tostring(ssl, "#.####")
    else if recentBOS and hasUnmitigatedOB
        if ms.orderFlow == "BULLISH"
            if close > nearestOBLevel
                expectation := "📉 Pullback expected to bullish OB at " + str.tostring(nearestOBLevel, "#.####") + " before continuation up"
            else
                expectation := "🚀 Price approaching bullish OB. Long entry opportunity near " + str.tostring(nearestOBLevel, "#.####")
        else
            if close < nearestOBLevel
                expectation := "📈 Retracement expected to bearish OB at " + str.tostring(nearestOBLevel, "#.####") + " before continuation down"
            else
                expectation := "💥 Price approaching bearish OB. Short entry opportunity near " + str.tostring(nearestOBLevel, "#.####")
    else if ms.choch
        expectation := "🔄 Character change! Old " + (ms.orderFlow == "BULLISH" ? "resistance" : "support") + " becomes " + (ms.orderFlow == "BULLISH" ? "support" : "resistance") + ". First retest pending"
    else if isPremium and ms.orderFlow == "BEARISH"
        expectation := "📍 In premium zone + bearish bias. Seeking sell setups. Target: " + str.tostring(equilibrium, "#.####")
    else if isDiscount and ms.orderFlow == "BULLISH"
        expectation := "📍 In discount zone + bullish bias. Seeking buy setups. Target: " + str.tostring(equilibrium, "#.####")
    else if inKillzone
        if isLondonKZ
            expectation := "⏰ London KZ active. Expect sweep of Asian high/low at " + str.tostring(bsl, "#.####") + "/" + str.tostring(ssl, "#.####")
        else if isNYAMKZ
            if ms.orderFlow == "NEUTRAL"
                expectation := "⏰ NY AM KZ: Awaiting directional commitment. Monitor " + str.tostring(bsl, "#.####") + " and " + str.tostring(ssl, "#.####")
            else
                expectation := "⏰ NY AM KZ + " + ms.orderFlow + " bias. Expect acceleration " + (ms.orderFlow == "BULLISH" ? "higher" : "lower")
        else if isNYPMKZ
            expectation := "⏰ NY PM KZ: Position squaring time. Fade moves without strong volume"
    else if isEquilibrium
        nearBSL = math.abs(close - bsl) < atr14 * 2
        nearSSL = math.abs(close - ssl) < atr14 * 2
        if nearBSL
            expectation := "⚖️ At equilibrium near BSL. Rejection = Short, Break = Long targeting " + str.tostring(bsl + atr14 * 2, "#.####")
        else if nearSSL
            expectation := "⚖️ At equilibrium near SSL. Rejection = Long, Break = Short targeting " + str.tostring(ssl - atr14 * 2, "#.####")
        else
            expectation := "⚖️ At equilibrium. Wait for directional break with volume > " + str.tostring(math.round(volumeSMA20 * 1.5), "#")
    else
        if array.size(elements) == 0
            expectation := "🔍 No quality setups detected. Wait for clear structure break or liquidity sweep"
        else
            expectation := "📊 Monitoring " + str.tostring(array.size(elements)) + " elements. Next key level: " + 
                          (close > equilibrium ? str.tostring(bsl, "#.####") + " (BSL)" : str.tostring(ssl, "#.####") + " (SSL)")
    narrative.nextExpectation := expectation
    narrative.causalSequence := sequence
getEnhancedMarketNarrative() =>
    buildMarketNarrative()
    fullNarrative = "📍 " + narrative.currentPhase + "\n"
    fullNarrative += "🎯 Bias: " + ms.orderFlow
    if htfBias == ms.orderFlow and ms.orderFlow != "NEUTRAL"
        fullNarrative += " ✅ HTF Aligned"
    else if htfBias != "NEUTRAL" and ms.orderFlow != "NEUTRAL" and htfBias != ms.orderFlow
        fullNarrative += " ⚠️ HTF Conflict"
    fullNarrative += "\n"
    marketPosition = isPremium ? "Premium" : isDiscount ? "Discount" : "Equilibrium"
    fullNarrative += "📊 Zone: " + marketPosition
    if isPremium or isDiscount
        fullNarrative += " (" + str.tostring(math.round(math.abs(close - equilibrium) / atr14), "#") + " ATR from EQ)"
    fullNarrative += "\n"
    if narrative.causalSequence != ""
        fullNarrative += "🔗 Events: " + narrative.causalSequence + "\n"
    fullNarrative += "🎯 " + narrative.nextExpectation + "\n"
    if array.size(elements) > 0
        lastE = safeGet(elements, array.size(elements) - 1)
        if not na(lastE) and not lastE.mitigated
            score = calculateTradeScore(lastE)
            if score.total >= 60
                suggestion = generateNarrativeTradeSuggestion(lastE)
                if suggestion.action != ""
                    fullNarrative += "⚡ Setup Ready: " + suggestion.action + " RR:" + str.tostring(suggestion.riskReward, "#.#")                    
    fullNarrative
classifyElement(element) =>
    if na(element)
        element
    else
        age = bar_index - element.endBar
        causedBullishBOS = false
        causedBearishBOS = false        
        if ms.bullishBOS and ms.lastBOSBar > element.endBar and ms.lastBOSBar - element.endBar <= 10
            causedBullishBOS := true
            element.causedBOS := true
        if ms.bearishBOS and ms.lastBOSBar > element.endBar and ms.lastBOSBar - element.endBar <= 10
            causedBearishBOS := true
            element.causedBOS := true
        liquiditySweep = (bslSwept or sslSwept) and bar_index - element.endBar <= 5
        highVolume = false
        if element.endBar >= bar_index - maxHistoryBars
            idx = bar_index - element.endBar
            if idx >= 0 and idx < 500
                highVolume := volume[idx] > volumeSMA20[idx] * 1.5
        if causedBullishBOS or causedBearishBOS
            element.elementType := "Order Block"
            element.narrativeRole := causedBullishBOS ? "Bullish OB (caused BOS)" : "Bearish OB (caused BOS)"
            element.strength := 0.9
        else if liquiditySweep
            element.elementType := "Trap Zone"
            element.narrativeRole := "Liquidity Trap"
            element.strength := 0.8
        else if highVolume and element.inLogicalArea
            element.elementType := "Order Block"
            element.narrativeRole := ms.orderFlow + " OB (high volume)"
            element.strength := 0.7
        else if age > 5
            element.elementType := "S/R Zone"
            element.narrativeRole := "Support/Resistance"
            element.strength := 0.5           
        element  
determineAdvancedICTType(e, breakDirection) =>
    ictType = "Pending"
    strength = 0.5
    narrativeRole = ""
    if na(e)
        [ictType, strength, narrativeRole]
    else
        barsSinceElement = bar_index - e.endBar
        if barsSinceElement < 2
            [ictType, strength, narrativeRole]
        else
            isStrongMove = math.abs(close - close[2]) > atr14 * 1.5
            hasVolume = volume > volumeSMA20 * 1.3 and volume[1] > volumeSMA20 * 1.2
            hasBOS = ms.bullishBOS or ms.bearishBOS
            hasExternalBOS = ms.externalBullishBOS or ms.externalBearishBOS
            hasLiquiditySweep = bslSwept or sslSwept
            isPivot = not na(internalHigh) or not na(internalLow)
            causedStructuralBreak = false
            if array.size(elements) > 0
                for i = array.size(elements) - 1 to math.max(0, array.size(elements) - 5)
                    elem = safeGet(elements, i)
                    if not na(elem) and elem.endBar >= ms.lastBOSBar - 5 and elem.endBar <= ms.lastBOSBar
                        causedStructuralBreak := true
                        elem.causedBOS := true
                        break
            if breakDirection == "up"
                if hasExternalBOS and hasVolume and isStrongMove and e.insideKillzone
                    ictType := "Type 1"
                    strength := math.min(volume / volumeSMA20 * 0.7, 1.0)
                    narrativeRole := "Momentum continuation after external BOS"
                else if hasLiquiditySweep and close < open
                    ictType := "Type 2"
                    strength := 0.8
                    narrativeRole := "Liquidity grab and reversal"
                else if isPivot or (rsi14 > 70 and not hasVolume)
                    ictType := "Type 3"
                    strength := 0.6
                    narrativeRole := "Potential reversal pivot"
                else
                    ictType := "Type 3"
                    strength := 0.5
                    narrativeRole := "Minor structure"
            else
                if hasExternalBOS and hasVolume and isStrongMove and e.insideKillzone
                    ictType := "Type 1"
                    strength := math.min(volume / volumeSMA20 * 0.7, 1.0)
                    narrativeRole := "Momentum continuation after external BOS"
                else if hasLiquiditySweep and close > open
                    ictType := "Type 2"
                    strength := 0.8
                    narrativeRole := "Liquidity grab and reversal"
                else if isPivot or (rsi14 < 30 and not hasVolume)
                    ictType := "Type 3"
                    strength := 0.6
                    narrativeRole := "Potential reversal pivot"
                else
                    ictType := "Type 3"
                    strength := 0.5
                    narrativeRole := "Minor structure"
            [ictType, math.min(strength, 1.0), narrativeRole]
assessQuality(element) =>
    if na(element)
        0.0
    else
        score = 0.0
        if element.insideKillzone
            score += 20
        else if isLondonSession or isNYSession
            score += 10  
        if (ms.orderFlow == "BULLISH" and isDiscount) or (ms.orderFlow == "BEARISH" and isPremium)
            score += 25
            element.inLogicalArea := true
        else if isEquilibrium
            score += 10
        if volume > volumeSMA20 * 1.5
            score += 20
        fvgPresent = math.abs(high[1] - low[2]) > atr14 * 0.5 or math.abs(low[1] - high[2]) > atr14 * 0.5
        if fvgPresent and volume[1] > volumeSMA20 * 1.3
            score += 20
            element.hasHighVolumeFVG := true
        if htfBias == ms.orderFlow
            score += 15  
        math.min(score, 100.0)
getIPDAWindows() =>
    windows = array.new<IPDAWindow>()
    array.push(windows, IPDAWindow.new("Asian Range", "0000", "0300", not na(time(timeframe.period, "0000-0300", "UTC"))))
    array.push(windows, IPDAWindow.new("London Open", "0800", "0900", not na(time(timeframe.period, "0800-0900", "UTC"))))
    array.push(windows, IPDAWindow.new("NY Open", "1330", "1430", not na(time(timeframe.period, "1330-1430", "UTC"))))
// ========================================
 // DASHBOARD FUNCTIONS
// ========================================
cell(t, col, row, txt, txtColor, bgColor, trans) =>
    table.cell(t, col, row, txt, text_color=txtColor, bgcolor=color.new(bgColor, trans), text_size=size.small)
drawDashboard(t) =>
    table result = na
    if not showDashboard
        if not na(t)
            table.delete(t)
    else
        if not na(t)
            table.delete(t)
        type1Count = 0
        type2Count = 0
        type3Count = 0
        pendingCount = 0
        mitigatedCount = 0
        unmitigatedCount = 0
        highQualityCount = 0
        if array.size(elements) > 0
            for i = 0 to math.min(array.size(elements) - 1, 50)
                e = safeGet(elements, i)
                if not na(e)
                    if e.elementType == "Order Block"
                        type1Count += 1
                    else if e.elementType == "Trap Zone"
                        type2Count += 1
                    else if e.elementType == "Reversal Zone" or e.elementType == "S/R Zone"
                        type3Count += 1
                    else if e.elementType == "Pending"
                        pendingCount += 1
                    if e.mitigated
                        mitigatedCount += 1
                    else
                        unmitigatedCount += 1                       
                    if e.qualityScore >= 65
                        highQualityCount += 1       
        smFlow = calculateSmartMoneyFlow()
        int base_rows = 10
        int medium_extra = 5
        int large_extra = 15
        int buffer = 5
        int footer_row = 1
        int max_rows = (dashboardSize == "Large" ? base_rows + medium_extra + large_extra : dashboardSize == "Medium" ? base_rows + medium_extra : base_rows) + buffer + footer_row
        pos = dashboardPosition == "Top Left" ? position.top_left : dashboardPosition == "Top Right" ? position.top_right : dashboardPosition == "Bottom Left" ? position.bottom_left : position.bottom_right
        table currentTable = table.new(pos, 4, max_rows, border_width = 1)       
        row = 0
        cell(currentTable, 0, row, "Advanced ICT Theory", color.white, color.black, dashboardTransparency)
        table.merge_cells(currentTable, 0, row, 3, row)
        row += 1
        cell(currentTable, 0, row, "ICT METRICS & BREAKDOWN", color.white, #2A2E39, dashboardTransparency)
        table.merge_cells(currentTable, 0, row, 3, row)
        row += 1
        cell(currentTable, 0, row, "Total Elements", color.white, color.gray, dashboardTransparency)
        cell(currentTable, 1, row, str.tostring(array.size(elements)), color.white, color.gray, dashboardTransparency)
        cell(currentTable, 2, row, "High Quality", color.green, color.gray, dashboardTransparency)
        cell(currentTable, 3, row, str.tostring(highQualityCount), color.white, color.gray, dashboardTransparency)
        row += 1       
        cell(currentTable, 0, row, "Pending", color.yellow, color.gray, dashboardTransparency)
        cell(currentTable, 1, row, str.tostring(pendingCount), color.white, color.gray, dashboardTransparency)
        cell(currentTable, 2, row, "Unmitigated", #00E676, color.gray, dashboardTransparency)
        cell(currentTable, 3, row, str.tostring(unmitigatedCount - pendingCount), color.white, color.gray, dashboardTransparency)
        row += 1       
        cell(currentTable, 0, row, "Mitigated", #FF5252, color.gray, dashboardTransparency)
        cell(currentTable, 1, row, str.tostring(mitigatedCount), color.white, color.gray, dashboardTransparency)
        cell(currentTable, 2, row, "Order Blocks", type1BoxColor, color.gray, dashboardTransparency)
        cell(currentTable, 3, row, str.tostring(type1Count), color.white, color.gray, dashboardTransparency)
        row += 1        
        cell(currentTable, 0, row, "Trap Zones", type2BoxColor, color.gray, dashboardTransparency)
        cell(currentTable, 1, row, str.tostring(type2Count), color.white, color.gray, dashboardTransparency)
        cell(currentTable, 2, row, "Reversal/S&R", type3BoxColor, color.gray, dashboardTransparency)
        cell(currentTable, 3, row, str.tostring(type3Count), color.white, color.gray, dashboardTransparency)
        row += 1       
        cell(currentTable, 0, row, "FVGs Active", fvgBoxColor, color.gray, dashboardTransparency)
        cell(currentTable, 1, row, str.tostring(array.size(fvgPatterns)), color.white, color.gray, dashboardTransparency)
        row += 1
        cell(currentTable, 0, row, "STRUCTURE & MARKET CONTEXT", color.white, #2A2E39, dashboardTransparency)
        table.merge_cells(currentTable, 0, row, 3, row)
        row += 1
        cell(currentTable, 0, row, "Order Flow", color.white, color.gray, dashboardTransparency)
        flowColor = ms.orderFlow == "BULLISH" ? color.green : ms.orderFlow == "BEARISH" ? color.red : color.yellow
        cell(currentTable, 1, row, ms.orderFlow, flowColor, color.gray, dashboardTransparency)
        cell(currentTable, 2, row, "Last BOS", color.white, color.gray, dashboardTransparency)
        bosText = ms.externalBullishBOS ? "External Bull" : ms.externalBearishBOS ? "External Bear" : ms.internalBullishBOS ? "Internal Bull" : ms.internalBearishBOS ? "Internal Bear" : "None"
        cell(currentTable, 3, row, bosText, color.white, color.gray, dashboardTransparency)
        row += 1        
        cell(currentTable, 0, row, "CHoCH Active", color.white, color.gray, dashboardTransparency)
        chochColor = ms.choch ? color.orange : color.gray
        cell(currentTable, 1, row, ms.choch ? "YES" : "NO", chochColor, color.gray, dashboardTransparency)
        cell(currentTable, 2, row, "HTF Bias", color.white, color.gray, dashboardTransparency)
        htfColor = htfBias == "BULLISH" ? color.green : htfBias == "BEARISH" ? color.red : color.yellow
        alignText = htfBias == ms.orderFlow ? " ✓" : " ✗"
        cell(currentTable, 3, row, htfBias + alignText, htfColor, color.gray, dashboardTransparency)
        row += 1       
        cell(currentTable, 0, row, "Market State", color.white, color.gray, dashboardTransparency)
        stateText = isPremium ? "PREMIUM" : isDiscount ? "DISCOUNT" : "EQUILIBRIUM"
        stateColor = isPremium ? color.red : isDiscount ? color.green : color.yellow
        cell(currentTable, 1, row, stateText, stateColor, color.gray, dashboardTransparency)
        cell(currentTable, 2, row, "Liquidity", color.white, color.gray, dashboardTransparency)
        liqText = bslSwept ? "BSL Swept" : sslSwept ? "SSL Swept" : "Protected"
        liqColor = bslSwept or sslSwept ? color.orange : color.white
        cell(currentTable, 3, row, liqText, liqColor, color.gray, dashboardTransparency)
        row += 1
        if dashboardSize == "Medium" or dashboardSize == "Large"
            cell(currentTable, 0, row, "SMART MONEY FLOW", color.white, #2A2E39, dashboardTransparency)
            table.merge_cells(currentTable, 0, row, 3, row)
            row += 1
            cell(currentTable, 0, row, "Net Flow", color.white, color.gray, dashboardTransparency)
            flowColor_SM = smFlow.netFlow > 0 ? color.green : smFlow.netFlow < 0 ? color.red : color.yellow
            cell(currentTable, 1, row, smFlow.bias, flowColor, color.gray, dashboardTransparency)
            row += 1
            cell(currentTable, 2, row, "Buy Volume", color.green, color.gray, dashboardTransparency)
            cell(currentTable, 3, row, str.tostring(math.round(smFlow.buyVolume)), color.white, color.gray, dashboardTransparency)            
            cell(currentTable, 0, row, "Sell Volume", color.red, color.gray, dashboardTransparency)
            cell(currentTable, 1, row, str.tostring(math.round(smFlow.sellVolume)), color.white, color.gray, dashboardTransparency)
            row += 1
        if dashboardSize == "Large"
            cell(currentTable, 0, row, "KEY GUIDE", color.white, #2A2E39, dashboardTransparency)
            table.merge_cells(currentTable, 0, row, 3, row)
            row += 1
            cell(currentTable, 0, row, "PATTERN TYPES", color.white, #2A2E39, dashboardTransparency)
            table.merge_cells(currentTable, 0, row, 3, row)
            row += 1
            cell(currentTable, 0, row, "━━━", type1BoxColor, color.gray, dashboardTransparency)
            cell(currentTable, 1, row, "Order Block (OB)", color.white, color.gray, dashboardTransparency)
            cell(currentTable, 2, row, "Institutional orders", color.black, color.gray, dashboardTransparency)
            cell(currentTable, 3, row, "High probability zone", color.white, color.gray, dashboardTransparency)
            row += 1
            cell(currentTable, 0, row, "━━━", type2BoxColor, color.gray, dashboardTransparency)
            cell(currentTable, 1, row, "Trap Zone", color.white, color.gray, dashboardTransparency)
            cell(currentTable, 2, row, "Liquidity grab area", color.black, color.gray, dashboardTransparency)
            cell(currentTable, 3, row, "Liquidity trap", color.white, color.gray, dashboardTransparency)
            row += 1
            cell(currentTable, 0, row, "━━━", type3BoxColor, color.gray, dashboardTransparency)
            cell(currentTable, 1, row, "Reversal/S&R", color.white, color.gray, dashboardTransparency)
            cell(currentTable, 2, row, "Key support/resistance", color.black, color.gray, dashboardTransparency)
            cell(currentTable, 3, row, "Reversal point", color.white, color.gray, dashboardTransparency)
            row += 1
            cell(currentTable, 0, row, "━━━", fvgBoxColor, color.gray, dashboardTransparency)
            cell(currentTable, 1, row, "FVG", color.white, color.gray, dashboardTransparency)
            cell(currentTable, 2, row, "Fair Value Gap", color.black, color.gray, dashboardTransparency)
            cell(currentTable, 3, row, "Gap opportunity", color.white, color.gray, dashboardTransparency)
            row += 1
            cell(currentTable, 0, row, "━━━", color.yellow, color.gray, dashboardTransparency)
            cell(currentTable, 1, row, "Pending", color.white, color.gray, dashboardTransparency)
            cell(currentTable, 2, row, "Unclassified zone", color.black, color.gray, dashboardTransparency)
            cell(currentTable, 3, row, "Await classification", color.white, color.gray, dashboardTransparency)
            row += 1
            cell(currentTable, 0, row, "STRUCTURE", color.white, #2A2E39, dashboardTransparency)
            table.merge_cells(currentTable, 0, row, 3, row)
            row += 1
            cell(currentTable, 0, row, "━━━", color.green, color.gray, dashboardTransparency)
            cell(currentTable, 1, row, "eBOS", color.white, color.gray, dashboardTransparency)
            cell(currentTable, 2, row, "External Break of Structure", color.black, color.gray, dashboardTransparency)
            cell(currentTable, 3, row, "Major break", color.white, color.gray, dashboardTransparency)
            row += 1
            cell(currentTable, 0, row, "━━━", color.orange, color.gray, dashboardTransparency)
            cell(currentTable, 1, row, "CHoCH", color.white, color.gray, dashboardTransparency)
            cell(currentTable, 2, row, "Change of Character", color.black, color.gray, dashboardTransparency)
            cell(currentTable, 3, row, "Trend change", color.white, color.gray, dashboardTransparency)
            row += 1
            cell(currentTable, 0, row, "SPECIAL LEVELS", color.white, #2A2E39, dashboardTransparency)
            table.merge_cells(currentTable, 0, row, 3, row)
            row += 1          
            cell(currentTable, 0, row, "━━━", originColor, color.gray, dashboardTransparency)
            cell(currentTable, 1, row, "Origin", color.white, color.gray, dashboardTransparency)
            cell(currentTable, 2, row, "Primary bias level", color.black, color.gray, dashboardTransparency)
            cell(currentTable, 3, row, "Bias origin", color.white, color.gray, dashboardTransparency)
            row += 1           
            cell(currentTable, 0, row, "━━━", mitigationColor, color.gray, dashboardTransparency)
            cell(currentTable, 1, row, "Mit", color.white, color.gray, dashboardTransparency)
            cell(currentTable, 2, row, "Mitigation level", color.black, color.gray, dashboardTransparency)
            cell(currentTable, 3, row, "Mitigation point", color.white, color.gray, dashboardTransparency)
            row += 1           
            cell(currentTable, 0, row, "━━━", color.blue, color.gray, dashboardTransparency)
            cell(currentTable, 1, row, "BSL", color.white, color.gray, dashboardTransparency)
            cell(currentTable, 2, row, "Buy-side liquidity", color.black, color.gray, dashboardTransparency)
            cell(currentTable, 3, row, "Buy liquidity", color.white, color.gray, dashboardTransparency)
            row += 1
            cell(currentTable, 0, row, "━━━", color.red, color.gray, dashboardTransparency)
            cell(currentTable, 1, row, "SSL", color.white, color.gray, dashboardTransparency)
            cell(currentTable, 2, row, "Sell-side liquidity", color.black, color.gray, dashboardTransparency)
            cell(currentTable, 3, row, "Sell liquidity", color.white, color.gray, dashboardTransparency)
            row += 1           
            cell(currentTable, 0, row, "GRADES", color.white, #2A2E39, dashboardTransparency)
            table.merge_cells(currentTable, 0, row, 3, row)
            row += 1
            cell(currentTable, 0, row, "[A+]", color.green, color.gray, dashboardTransparency)
            cell(currentTable, 1, row, "80%+", color.white, color.gray, dashboardTransparency)
            cell(currentTable, 2, row, "High probability", color.black, color.gray, dashboardTransparency)
            cell(currentTable, 3, row, "Top grade", color.white, color.gray, dashboardTransparency)
            row += 1
            cell(currentTable, 0, row, "[B]", color.yellow, color.gray, dashboardTransparency)
            cell(currentTable, 1, row, "60%+", color.white, color.gray, dashboardTransparency)
            cell(currentTable, 2, row, "Moderate quality", color.black, color.gray, dashboardTransparency)
            cell(currentTable, 3, row, "Medium grade", color.white, color.gray, dashboardTransparency)
            row += 1
            cell(currentTable, 0, row, "[C]", color.red, color.gray, dashboardTransparency)
            cell(currentTable, 1, row, "<60%", color.white, color.gray, dashboardTransparency)
            cell(currentTable, 2, row, "Low quality", color.black, color.gray, dashboardTransparency)
            cell(currentTable, 3, row, "Low grade", color.white, color.gray, dashboardTransparency)
            row += 1
        footerRowIndex = max_rows - 1
        cell(currentTable, 0, footerRowIndex, "💎 ⚡ Dskyz (DAFE) Trading Systems 💎", color.new(#FFD700, 0), color.new(#1A1A2E, 0), math.max(0, dashboardTransparency - 20))
        table.merge_cells(currentTable, 0, footerRowIndex, 3, footerRowIndex)
        result := currentTable
    result
drawNarrativeDashboard(t) =>
    table result = na
    if not showNarrativeDashboard
        if not na(t)
            table.delete(t)
    else
        if not na(t)
            table.delete(t)
        int max_rows = 20
        pos = narrativeDashboardPosition == "Top Left" ? position.top_left : narrativeDashboardPosition == "Top Right" ? position.top_right : narrativeDashboardPosition == "Bottom Left" ? position.bottom_left : position.bottom_right
        table currentTable = table.new(pos, 6, max_rows, border_width = 1, border_color = color.new(color.silver, 70), bgcolor = color.new(#1e222d, narrativeTransparency))
        dc_white = color.white
        dc_gray = color.silver
        dc_green = color.green
        dc_red = color.red
        dc_gold = color.yellow
        dc_orange = color.orange
        dc_aqua = color.aqua
        bg_header = color.new(color.black, 40)
        bg_section = color.new(color.gray, 80)
        bg_alt_row = color.new(color.gray, 90)
        header_size = size.normal
        value_size = size.small
        label_size = size.tiny
        current_row = 0
        table.merge_cells(currentTable, 0, current_row, 5, current_row)
        cell(currentTable, 0, current_row, "📖 Narrative Dashboard", dc_gold, bg_header, 0)
        current_row += 1
        table.merge_cells(currentTable, 0, current_row, 5, current_row)
        cell(currentTable, 0, current_row, "═══ 📊 RECENT ELEMENTS ═══", dc_gold, bg_section, 0)
        current_row += 1
        cell(currentTable, 0, current_row, "ID", dc_white, bg_header, 0)
        cell(currentTable, 1, current_row, "Type", dc_white, bg_header, 0)
        cell(currentTable, 2, current_row, "Role", dc_white, bg_header, 0)
        cell(currentTable, 3, current_row, "Quality", dc_white, bg_header, 0)
        cell(currentTable, 4, current_row, "Score", dc_white, bg_header, 0)
        cell(currentTable, 5, current_row, "Price", dc_white, bg_header, 0)
        current_row += 1
        displayCount = 0
        if array.size(elements) > 0
            for i = array.size(elements) - 1 to math.max(0, array.size(elements) - 10)
                if displayCount >= 5 or current_row >= max_rows - 2  // Prevent overflow
                    break
                e = safeGet(elements, i)
                if na(e) or e.qualityScore < minQualityThreshold
                    continue
                id = "#" + str.tostring(array.size(elements) - i)
                typ = switch e.elementType
                    "Type 1" => "T1"
                    "Type 2" => "T2"
                    "Type 3" => "T3"
                    => "Pnd"
                role = e.narrativeRole != "" ? e.narrativeRole : "Pending analysis"
                quality = str.tostring(math.round(e.qualityScore)) + "%"
                score = calculateTradeScore(e)
                scoreText = score.grade
                scoreColor = score.total >= 75 ? dc_green : score.total >= 55 ? dc_gold : dc_red
                emoji = score.total >= 75 ? "🌟" : score.total >= 55 ? "✅" : "⚠️"
                price = str.tostring((e.high + e.low)/2, "#.####")
                bg_row = current_row % 2 == 0 ? bg_alt_row : color.new(#000000, 100)
                cell(currentTable, 0, current_row, id, dc_white, bg_row, 0)
                cell(currentTable, 1, current_row, typ, dc_white, bg_row, 0)
                cell(currentTable, 2, current_row, role, dc_white, bg_row, 0)
                cell(currentTable, 3, current_row, quality, dc_aqua, bg_row, 0)
                cell(currentTable, 4, current_row, emoji + " " + scoreText, scoreColor, bg_row, 0)
                cell(currentTable, 5, current_row, price, dc_white, bg_row, 0)
                current_row += 1
                displayCount += 1
        if displayCount == 0 and current_row < max_rows
            table.merge_cells(currentTable, 0, current_row, 5, current_row)
            cell(currentTable, 0, current_row, "⚠️ No quality elements detected", dc_orange, color.new(color.gray, 0), 0)
            current_row += 1
        if current_row < max_rows
            table.merge_cells(currentTable, 0, current_row, 5, current_row)
            cell(currentTable, 0, current_row, "═══ 🔮 MARKET NARRATIVE ═══", dc_gold, bg_section, 0)
            current_row += 1
            narrativeText = getEnhancedMarketNarrative()
            lines = str.split(narrativeText, "\n")
            for i = 0 to math.min(array.size(lines) - 1, max_rows - current_row - 1)
                table.merge_cells(currentTable, 0, current_row, 5, current_row)
                lineColor = str.contains(array.get(lines, i), "High-Quality") ? dc_green : str.contains(array.get(lines, i), "Phase") ? dc_aqua : dc_white
                cell(currentTable, 0, current_row, array.get(lines, i), lineColor, color.new(#000000, 100), 0)
                current_row += 1
        result := currentTable
    result
// ========================================
// VALIDATION
// ========================================
isValidICTElement() =>
    if bar_index <= minDisplacementCandles or bar_index - lastDetectionBar < 5
        false
    else
        parentHigh = high[minDisplacementCandles]
        parentLow = low[minDisplacementCandles]
        allInside = true
        for i = 0 to minDisplacementCandles - 1
            if high[i] > parentHigh or low[i] < parentLow
                allInside := false
                break  
        fvgGap = math.abs(high[1] - low[minDisplacementCandles]) > atr14 * 0.3 or 
                 math.abs(low[1] - high[minDisplacementCandles]) > atr14 * 0.3   
        elementRange = parentHigh - parentLow
        rangeQuality = elementRange > avgRange20 * minElementSize and elementRange > atr14 * minElementSize
        volumeQuality = volume[minDisplacementCandles] > volumeSMA10 * 0.8
        recentElements = 0
        if array.size(elements) > 0
            for i = array.size(elements) - 1 to math.max(0, array.size(elements) - 5)
                e = safeGet(elements, i)
                if not na(e) and bar_index - e.endBar < 20
                    recentElements += 1  
        notTooMany = recentElements < 3
        htfValid = not htfAlignmentRequired or htfBias != "NEUTRAL"
        (allInside or fvgGap) and rangeQuality and volumeQuality and notTooMany and htfValid
// ========================================
 // MAIN DETECTION
// ========================================
detectAdvancedStructure()
if isValidICTElement() and barstate.isconfirmed
    newElement = ICTElement.new()
    newElement.high := high[minDisplacementCandles]
    newElement.low := low[minDisplacementCandles]
    newElement.startBar := bar_index - minDisplacementCandles
    newElement.endBar := bar_index
    newElement.mitigated := false
    newElement.timeframe := timeframe.period
    newElement.insideKillzone := inKillzone
    newElement.nearLiquidity := math.abs(newElement.high - bsl) < atr14 * 0.5 or math.abs(newElement.low - ssl) < atr14 * 0.5
    newElement.elementType := "Pending"
    newElement.narrativeRole := "Awaiting classification"
    newElement.qualityScore := assessQuality(newElement)
    if array.size(elements) < maxElementsToDisplay * 3
        array.push(elements, newElement)
        lastDetectionBar := bar_index
currentFVG = detectFVG()
if currentFVG.detected and array.size(fvgPatterns) < 20
    array.push(fvgPatterns, currentFVG)
// ========================================
// CLEANUP AND MITIGATION  
// ========================================
if barstate.isconfirmed
    if array.size(fvgPatterns) > 0
        for i = array.size(fvgPatterns) - 1 to 0
            fvg = safeGet(fvgPatterns, i)
            if not na(fvg) and bar_index - fvg.startBar > 50
                if not na(fvg.fvgBox)
                    box.delete(fvg.fvgBox)
                if not na(fvg.fvgLine)
                    line.delete(fvg.fvgLine)
                if not na(fvg.fvgLabel)
                    label.delete(fvg.fvgLabel)
                array.remove(fvgPatterns, i)
    toRemove = array.new<int>()
    if array.size(elements) > 0
        for i = 0 to math.min(array.size(elements) - 1, 50)
            e = safeGet(elements, i)
            if not na(e)
                age = bar_index - e.startBar
                if e.elementType == "Pending" and age >= 2 and age < pendingTimeout
                    causedBullishBOS = ms.bullishBOS and ms.lastBOSBar > e.endBar and ms.lastBOSBar - e.endBar <= 10
                    causedBearishBOS = ms.bearishBOS and ms.lastBOSBar > e.endBar and ms.lastBOSBar - e.endBar <= 10
                    if causedBullishBOS or causedBearishBOS
                        e.causedBOS := true
                        e.elementType := "Order Block"
                        e.narrativeRole := causedBullishBOS ? "Bullish OB (caused BOS)" : "Bearish OB (caused BOS)"
                        e.strength := 0.9
                    else if (bslSwept or sslSwept) and bar_index - e.endBar <= 5
                        e.elementType := "Trap Zone"
                        e.narrativeRole := "Liquidity Trap"
                        e.strength := 0.8
                    else if volume > volumeSMA20 * 1.5 and e.inLogicalArea
                        e.elementType := "Order Block"
                        e.narrativeRole := ms.orderFlow + " OB (high volume)"
                        e.strength := 0.7
                    else if age > 5
                        e.elementType := "S/R Zone"
                        e.narrativeRole := "Support/Resistance"
                        e.strength := 0.5
                if age > brokenAgeThreshold or (e.elementType == "Pending" and age > pendingTimeout)
                    if not na(e.identificationBox)
                        box.delete(e.identificationBox)
                    if not na(e.patternLine)
                        line.delete(e.patternLine)
                    if not na(e.patternLabel)
                        label.delete(e.patternLabel)
                    array.push(toRemove, i)  
    if array.size(toRemove) > 0
        for i = array.size(toRemove) - 1 to 0
            idx = array.get(toRemove, i)
            if idx < array.size(elements)
                array.remove(elements, idx)
    if array.size(elements) > 0
        for i = array.size(elements) - 1 to math.max(0, array.size(elements) - 20)
            e = safeGet(elements, i)
            if not na(e) and not e.mitigated
                mitigated = false
                breakDirection = ""
                if mitigationMethod == 'Cross'
                    if high > e.high
                        mitigated := true
                        breakDirection := "up"
                    else if low < e.low
                        mitigated := true
                        breakDirection := "down"
                else
                    if close > e.high
                        mitigated := true
                        breakDirection := "up"
                    else if close < e.low
                        mitigated := true
                        breakDirection := "down"
                if mitigated
                    e.mitigated := true
                    e.mitigationBar := bar_index
                    if e.elementType == "Pending"
                        if (breakDirection == "up" and ms.orderFlow == "BEARISH") or (breakDirection == "down" and ms.orderFlow == "BULLISH")
                            e.elementType := "Reversal Zone"
                            e.narrativeRole := "Counter-trend mitigation"
                        else
                            e.elementType := "S/R Zone"
                            e.narrativeRole := "Support/Resistance break"
                    if (breakDirection == "up" and ms.bullishBOS) or (breakDirection == "down" and ms.bearishBOS)
                        if not na(currentOrigin)
                            if currentOrigin.isOrigin
                                currentOrigin.isOrigin := false
                        e.isOrigin := true
                        currentOrigin := e
                    if array.size(mitigationLevels) >= 3
                        removed = array.shift(mitigationLevels)
                    mitLevel = ICTElement.new()
                    mitLevel.high := breakDirection == "up" ? e.high : e.low
                    mitLevel.low := mitLevel.high
                    mitLevel.startBar := bar_index
                    mitLevel.isMitigation := true
                    mitLevel.strength := e.strength
                    array.push(mitigationLevels, mitLevel)
// ========================================
 // DRAWING ENGINE
// ========================================
clearVisuals() =>
    while array.size(elementLines) > 0
        l = array.pop(elementLines)
        if not na(l)
            line.delete(l)
    while array.size(elementLabels) > 0
        lbl = array.pop(elementLabels)
        if not na(lbl)
            label.delete(lbl)
    while array.size(structureLines) > 0
        l = array.pop(structureLines)
        if not na(l)
            line.delete(l)
    while array.size(structureLabels) > 0
        lbl = array.pop(structureLabels)
        if not na(lbl)
            label.delete(lbl)
if barstate.isconfirmed
    clearVisuals()
    dashboard := drawDashboard(dashboard)
    narrativeDashboard := drawNarrativeDashboard(narrativeDashboard)
    if array.size(elements) > 0
        bestElement = ICTElement.new()
        bestScore = 0.0
        for i = array.size(elements) - 1 to math.max(0, array.size(elements) - 10)
            e = safeGet(elements, i)
            if not na(e) and not e.mitigated and e.elementType != "Pending"
                score = calculateTradeScore(e)
                if score.total > bestScore
                    bestScore := score.total
                    bestElement := e
        if bestScore >= 60
            narrative.keySetup := bestElement.elementType + " [" + calculateTradeScore(bestElement).grade + "]"
        else
            narrative.keySetup := ""
    if showSessions
        if not na(asianBox)
            box.delete(asianBox)
        if not na(londonBox)
            box.delete(londonBox)
        if not na(nyBox)
            box.delete(nyBox)
        if isAsianSession
            asianBox := box.new(bar_index, high * 1.02, bar_index + 1, low * 0.98, bgcolor=color.new(color.blue, 95), border_color=na)
        if isLondonSession
            londonBox := box.new(bar_index, high * 1.02, bar_index + 1, low * 0.98, bgcolor=color.new(color.yellow, 95), border_color=na)
        if isNYSession
            nyBox := box.new(bar_index, high * 1.02, bar_index + 1, low * 0.98, bgcolor=color.new(color.green, 95), border_color=na)
    if showKillzones
        if not na(londonKZBox)
            box.delete(londonKZBox)
        if not na(nyamKZBox)
            box.delete(nyamKZBox)
        if not na(nypmKZBox)
            box.delete(nypmKZBox)
        if isLondonKZ
            londonKZBox := box.new(bar_index, high * 1.02, bar_index + 1, low * 0.98, bgcolor=color.new(color.red, 90), border_color=color.red, border_width=1)
        if isNYAMKZ
            nyamKZBox := box.new(bar_index, high * 1.02, bar_index + 1, low * 0.98, bgcolor=color.new(color.red, 90), border_color=color.red, border_width=1)
        if isNYPMKZ
            nypmKZBox := box.new(bar_index, high * 1.02, bar_index + 1, low * 0.98, bgcolor=color.new(color.red, 90), border_color=color.red, border_width=1)
    if showPremiumDiscount
        if not na(premiumBox)
            box.delete(premiumBox)
        if not na(discountBox)
            box.delete(discountBox)
        if not na(equilibriumBox)
            box.delete(equilibriumBox)
        premiumBox := box.new(bar_index - 20, pdHigh, bar_index, equilibrium + (pdHigh - equilibrium) * 0.5, bgcolor=color.new(color.red, 85), border_color=na)
        discountBox := box.new(bar_index - 20, equilibrium - (equilibrium - pdLow) * 0.5, bar_index, pdLow, bgcolor=color.new(color.green, 85), border_color=na)
        equilibriumBox := box.new(bar_index - 20, equilibrium + atr14 * 0.1, bar_index, equilibrium - atr14 * 0.1, bgcolor=color.new(color.yellow, 80), border_color=color.yellow)
    if showLiquidity
        if not bslSwept
            bslLine = line.new(bar_index - liquidityLookback, bsl, bar_index + 5, bsl, color=color.blue, style=line.style_solid, width=2)
            array.push(elementLines, bslLine)
            bslLabel = label.new(bar_index + 5, bsl, "BSL", color=color.new(color.blue, 100), style=label.style_label_left, textcolor=color.blue, size=size.small)
            array.push(elementLabels, bslLabel)
        else if showSweptLiquidity
            bslLine = line.new(bar_index - 5, bsl, bar_index, bsl, color=color.new(color.blue, 50), style=line.style_dotted, width=1)
            array.push(elementLines, bslLine)
            bslLabel = label.new(bar_index, bsl, "BSL ✓", color=color.new(color.blue, 100), style=label.style_label_left, textcolor=color.new(color.blue, 50), size=size.small)
            array.push(elementLabels, bslLabel)
        if not sslSwept
            sslLine = line.new(bar_index - liquidityLookback, ssl, bar_index + 5, ssl, color=color.red, style=line.style_solid, width=2)
            array.push(elementLines, sslLine)
            sslLabel = label.new(bar_index + 5, ssl, "SSL", color=color.new(color.red, 100), style=label.style_label_left, textcolor=color.red, size=size.small)
            array.push(elementLabels, sslLabel)
        else if showSweptLiquidity
            sslLine = line.new(bar_index - 5, ssl, bar_index, ssl, color=color.new(color.red, 50), style=line.style_dotted, width=1)
            array.push(elementLines, sslLine)
            sslLabel = label.new(bar_index, ssl, "SSL ✓", color=color.new(color.red, 100), style=label.style_label_left, textcolor=color.new(color.red, 50), size=size.small)
            array.push(elementLabels, sslLabel)
    if showPatternLines
        if ms.externalBullishBOS and bar_index - ms.lastBOSBar < 50
            bosLine = line.new(ms.lastBOSBar, ms.lastBOSPrice, bar_index, ms.lastBOSPrice, color=color.green, style=line.style_solid, width=patternLineWidth)
            array.push(structureLines, bosLine)
            bosLabel = label.new(bar_index + labelPosition, ms.lastBOSPrice, "eBOS", color=color.new(color.green, 100), style=label.style_label_left, textcolor=color.green, size=size.small)
            array.push(structureLabels, bosLabel)
        if ms.externalBearishBOS and bar_index - ms.lastBOSBar < 50
            bosLine = line.new(ms.lastBOSBar, ms.lastBOSPrice, bar_index, ms.lastBOSPrice, color=color.red, style=line.style_solid, width=patternLineWidth)
            array.push(structureLines, bosLine)
            bosLabel = label.new(bar_index + labelPosition, ms.lastBOSPrice, "eBOS", color=color.new(color.red, 100), style=label.style_label_left, textcolor=color.red, size=size.small)
            array.push(structureLabels, bosLabel)
        if ms.choch and bar_index - ms.lastBOSBar < 50
            chochLine = line.new(ms.lastBOSBar, ms.lastBOSPrice, bar_index, ms.lastBOSPrice, color=color.orange, style=line.style_solid, width=patternLineWidth + 1)
            array.push(structureLines, chochLine)
            chochLabel = label.new(bar_index + labelPosition, ms.lastBOSPrice, "CHoCH", color=color.new(color.orange, 100), style=label.style_label_left, textcolor=color.orange, size=size.normal)
            array.push(structureLabels, chochLabel)
    if array.size(fvgPatterns) > 0
        for i = array.size(fvgPatterns) - 1 to 0
            fvg = safeGet(fvgPatterns, i)
            if not na(fvg)
                if na(fvg.fvgBox) and showIdentificationBoxes
                    boxColor = fvg.isBullish ? color.new(fvgBoxColor, boxTransparency) : color.new(fvgBoxColor, boxTransparency)
                    fvg.fvgBox := box.new(fvg.startBar, fvg.high, bar_index, fvg.low, bgcolor=boxColor, border_color=fvgBoxColor, border_width=1)
                if na(fvg.fvgLine) and showPatternLines
                    midPoint = (fvg.high + fvg.low) / 2
                    fvg.fvgLine := line.new(fvg.startBar, midPoint, bar_index, midPoint, color=fvgBoxColor, style=line.style_dashed, width=1)
                    fvg.fvgLabel := label.new(bar_index + labelPosition, midPoint, "FVG", color=color.new(fvgBoxColor, 100), style=label.style_label_left, textcolor=fvgBoxColor, size=size.small)
                else
                    if not na(fvg.fvgBox)
                        box.set_right(fvg.fvgBox, bar_index)
                    if not na(fvg.fvgLine)
                        line.set_x2(fvg.fvgLine, bar_index)
                        if not na(fvg.fvgLabel)
                            label.set_x(fvg.fvgLabel, bar_index + labelPosition)
    if showElementLines and array.size(elements) > 0
        elementCount = 0
        for i = array.size(elements) - 1 to 0
            if elementCount >= maxElementsToDisplay
                break
            e = safeGet(elements, i)
            if na(e) or e.qualityScore < minQualityThreshold
                continue
            age = bar_index - e.startBar
            if age > brokenAgeThreshold
                continue
            lineColor = color.gray
            boxColor = color.gray
            if e.elementType == "Order Block"
                lineColor := type1BoxColor
                boxColor := color.new(type1BoxColor, boxTransparency)
            else if e.elementType == "Trap Zone"
                lineColor := type2BoxColor
                boxColor := color.new(type2BoxColor, boxTransparency)
            else if e.elementType == "Reversal Zone" or e.elementType == "S/R Zone"
                lineColor := type3BoxColor
                boxColor := color.new(type3BoxColor, boxTransparency)
            else if e.elementType == "Pending"
                lineColor := color.yellow
                boxColor := color.new(color.yellow, boxTransparency + 10)
            lineWidth = e.qualityScore >= 75 ? 3 : e.qualityScore >= 50 ? 2 : 1
            startBar = math.max(e.startBar, bar_index - 500)
            if showIdentificationBoxes and na(e.identificationBox) and e.elementType != "Pending"
                e.identificationBox := box.new(startBar, e.high, e.endBar, e.low, bgcolor=boxColor, border_color=lineColor, border_width=1)
            if showPatternLines and na(e.patternLine)
                midPoint = (e.high + e.low) / 2
                e.patternLine := line.new(e.endBar, midPoint, bar_index, midPoint, color=lineColor, style=line.style_solid, width=patternLineWidth)
                typeLabel = e.elementType == "Order Block" ? "OB" : e.elementType == "Trap Zone" ? "TRAP" : e.elementType == "Reversal Zone" ? "REV" : e.elementType == "S/R Zone" ? "S/R" : "PEND"
                score = calculateTradeScore(e)
                if showTradeScore and e.elementType != "Pending"
                    typeLabel += " [" + score.grade + "]"
                e.patternLabel := label.new(bar_index + labelPosition, midPoint, typeLabel, color=color.new(lineColor, 80), style=label.style_label_left, textcolor=lineColor, size=size.small)
            else if not na(e.patternLine)
                line.set_x2(e.patternLine, bar_index)
                if not na(e.patternLabel)
                    label.set_x(e.patternLabel, bar_index + labelPosition)
            elementCount += 1
    if showOriginLine and not na(currentOrigin)
        if currentOrigin.isOrigin
            if bar_index - currentOrigin.mitigationBar < brokenAgeThreshold
                originLine = line.new(math.max(currentOrigin.mitigationBar, bar_index - 500), currentOrigin.high, bar_index + labelPosition, currentOrigin.high, color=originColor, style=line.style_solid, width=originLineWidth)
                array.push(elementLines, originLine)
                originLabel = label.new(bar_index + labelPosition, currentOrigin.high, "Origin", color=color.new(originColor, 100), style=label.style_label_left, textcolor=originColor, size=size.small)
                array.push(elementLabels, originLabel)
    if showMitigationLevels and array.size(mitigationLevels) > 0
        mitDrawnCount = 0
        for i = 0 to math.min(array.size(mitigationLevels) - 1, 2)
            if mitDrawnCount >= 2
                break
            mit = safeGet(mitigationLevels, i)
            if not na(mit) and bar_index - mit.startBar < brokenAgeThreshold
                mitLine = line.new(math.max(mit.startBar, bar_index - 500), mit.high, bar_index + labelPosition, mit.high, color=mitigationColor, style=line.style_solid, width=mitigationLineWidth)
                array.push(elementLines, mitLine)
                mitLabel = label.new(bar_index + labelPosition, mit.high, "Mit " + str.tostring(mitDrawnCount + 1), color=color.new(mitigationColor, 100), style=label.style_label_left, textcolor=mitigationColor, size=size.small)
                array.push(elementLabels, mitLabel)
                mitDrawnCount += 1
    if showIPDA
        if ipdaAsianRange
            ipdaBox = box.new(bar_index, high * 1.001, bar_index + 1, low * 0.999, bgcolor=color.new(color.blue, 92), border_color=color.blue, border_width=1)
        if ipdaLondonOpen
            ipdaBox = box.new(bar_index, high * 1.001, bar_index + 1, low * 0.999, bgcolor=color.new(color.orange, 92), border_color=color.orange, border_width=1)
        if ipdaNYOpen
            ipdaBox = box.new(bar_index, high * 1.001, bar_index + 1, low * 0.999, bgcolor=color.new(color.purple, 92), border_color=color.purple, border_width=1)
while array.size(elements) > maxElementsToDisplay * 3
    removed = array.shift(elements)
    if not na(removed.identificationBox)
        box.delete(removed.identificationBox)
    if not na(removed.patternLine)
        line.delete(removed.patternLine)
    if not na(removed.patternLabel)
        label.delete(removed.patternLabel)

// ========================================
// 🛠 STRATEGY INPUTS (No-Repaint, HTF signal, Tick entry)
// ========================================
group_strategy = "🛠 Strategy Settings"
enableLong      = input.bool(true,  "Enable Long",      group=group_strategy)
enableShort     = input.bool(true,  "Enable Short",     group=group_strategy)
riskPct         = input.float(1.0,  "Position size (% equity)", minval=0.1, step=0.1, group=group_strategy, tooltip="Percent of equity used per trade")
atrSLmult       = input.float(1.5,  "ATR SL Multiplier",         minval=0.1, step=0.1, group=group_strategy)
tpRR            = input.float(1.5,  "Take Profit R multiple",    minval=0.5, step=0.1, group=group_strategy)
useLTFfilter    = input.bool(true,  "Use LTF Premium/Discount filter", group=group_strategy, tooltip="Longs require Discount; Shorts require Premium")

// ========================================
// 📈 HTF SIGNALS (computed on HTF, pulled down w/ lookahead_off → non-repainting)
// Break above/below previous swing window + trend filter (SMA20)
// ========================================
longCondHTF = request.security(syminfo.tickerid, htfTimeframe, (close > ta.sma(close, 20)) and (close > ta.highest(high[1], externalPivotLookback)), gaps=barmerge.gaps_off, lookahead=barmerge.lookahead_off)
shortCondHTF = request.security(syminfo.tickerid, htfTimeframe, (close < ta.sma(close, 20)) and (close < ta.lowest(low[1],  externalPivotLookback)), gaps=barmerge.gaps_off, lookahead=barmerge.lookahead_off)

// LTF filter: use premium/discount on current TF if enabled
ltfFilterLong  = not useLTFfilter or isDiscount
ltfFilterShort = not useLTFfilter or isPremium

// Bucket ID per HTF bar to make sure we at most 1 new entry per HTF bar
var int lastTradeBucket = na
htfBucket = request.security(syminfo.tickerid, htfTimeframe, bar_index, gaps=barmerge.gaps_off, lookahead=barmerge.lookahead_off)
canTradeThisBucket = na(lastTradeBucket) or (htfBucket != lastTradeBucket)

// ========================================
// ⚡ ENTRY LOGIC (tick entry, no repaint, 낙장불입)
// ========================================
if canTradeThisBucket and strategy.position_size == 0
    if enableLong and longCondHTF and ltfFilterLong
        strategy.entry("Long", strategy.long, qty=riskPct)
        lastTradeBucket := htfBucket
    if enableShort and shortCondHTF and ltfFilterShort
        strategy.entry("Short", strategy.short, qty=riskPct)
        lastTradeBucket := htfBucket

// ========================================
// 📤 RISK (static SL/TP based on ATR)
// ========================================
var float longStop = na
var float longLimit = na
var float shortStop = na
var float shortLimit = na

if strategy.position_size > 0
    longStop  := strategy.position_avg_price - atr14 * atrSLmult
    longLimit := strategy.position_avg_price + (strategy.position_avg_price - longStop) * tpRR
    strategy.exit("L-Exit", "Long", stop=longStop, limit=longLimit)

if strategy.position_size < 0
    shortStop  := strategy.position_avg_price + atr14 * atrSLmult
    shortLimit := strategy.position_avg_price - (shortStop - strategy.position_avg_price) * tpRR
    strategy.exit("S-Exit", "Short", stop=shortStop, limit=shortLimit)

// ========================================
 // ALERTS
// ========================================
if barstate.isconfirmed
    if array.size(elements) > 0
        for i = array.size(elements) - 1 to math.max(0, array.size(elements) - 5)
            e = safeGet(elements, i)
            if not na(e) and e.mitigated and e.mitigationBar == bar_index
                score = calculateTradeScore(e)
                if score.total >= 70
                    alertMessage = '🔥 HIGH QUALITY ICT SETUP!\n'
                    alertMessage += 'Type: ' + e.elementType + ' [' + score.grade + ']\n'
                    alertMessage += 'Price: ' + str.tostring(close, '#.####') + '\n'
                    alertMessage += 'Quality: ' + str.tostring(math.round(e.qualityScore)) + '%\n'
                    if e.causedBOS
                        alertMessage += '⚡ Caused BOS\n'
                    if e.insideKillzone
                        alertMessage += '🎯 In Killzone\n'
                    if e.nearLiquidity
                        alertMessage += '💧 Near Liquidity\n'
                    alert(alertMessage, alert.freq_once_per_bar)
                    break
    if ms.choch
        alert('⚠️ CHANGE OF CHARACTER!\n' + narrative.lastEvent, alert.freq_once_per_bar)
    if bslSwept or sslSwept
        sweepType = bslSwept ? 'Buy-side' : 'Sell-side'
        alert('💧 LIQUIDITY SWEEP!\n' + sweepType + ' liquidity swept.', alert.freq_once_per_bar)




이상이고, 필터는 가능한 적었으면 좋겠어 메인로직먼저 잡고 해당 로직 바탕으로 백테스트통해 기초 틀 파라미터 잡은 후에 필터 얹을 예정이니까 필터들은 우선 작업'고려'만 해주고 메인로직 아주 강력한놈으로 만들어줘
