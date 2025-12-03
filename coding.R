#需要的套件
library(readxl)
library(torch)
library(gstat)
library(dplyr)
library(tidyr)
library(sp)
library(raster)
library(ggplot2)
library(sf)
#載入資料
setwd("C:/Users/s0958/Downloads/資源所面試")
groundwater<-read_excel("地下水位.xlsx")
rain<-read_excel("降雨.xlsx")
pump<-read_excel("抽水量.xlsx")
town <- st_read("TOWN_MOI_1130718.shp")
yunlin<- town[town$COUNTYNAME == "雲林縣",]
yunlin <- st_transform(yunlin, CRS("+init=epsg:3826 +ellps=WGS84"))

##########
ggplot() +
  geom_sf(data = yunlin, fill = "white", color = "black") +
  geom_point(data = rain, aes(x = TWD97_X, y = TWD97_Y, color = year2016), alpha = 1,size = 3) +  # 根據降雨量改變點的大小和顏色
  scale_color_gradient(low = "blue", high = "yellow") +  # 顏色漸層
  theme_minimal()
#########

# 添加一個 type 欄位來區分站點類型
rain$type <- "降雨量測站"
groundwater$type <- "地下水位測站"
pump$type<- "抽水量測站"

# 合併兩個資料框
stations <- rbind(rain, groundwater,pump)

# 繪製地圖
ggplot() +
  geom_sf(data = yunlin, fill = "lightgrey", color = "black") +  # 雲林縣地圖
  geom_point(data = stations, aes(x = TWD97_X, y = TWD97_Y, color = type), size = 3, alpha = 0.6) +  # 根據站點類型設置顏色
  scale_color_manual(values = c("降雨量測站" = "blue", "地下水位測站" = "red","抽水量測站" = "green")) +  # 定義站點顏色
  guides(color = guide_legend(title = "測站類型")) +  # 自定義圖例標籤
  labs(title = "測站位置圖", 
       x = "經度",  # 設置 X 軸標題
       y = "緯\n度") +  # 設置 Y 軸標題
  theme_minimal()+
  theme(axis.title.y = element_text(angle = 0, vjust = 0.5))

#############################################

groundwater<-read_excel("地下水位.xlsx")
rain<-read_excel("降雨.xlsx")
pump<-read_excel("抽水量.xlsx")
head(rain) #降雨量
head(groundwater) #地下水位高度
head(pump)

#對降雨量空間插值
coordinates(rain) <- ~TWD97_X+TWD97_Y
proj4string(rain) <- CRS("+init=epsg:3826 +ellps=WGS84")
vgm_rain1 <- variogram(year2016 ~ 1, rain)
plot(vgm_rain1)
Var.rain1 <- vgm(psill=250000, "Wav", range=10000, nugget=0)
plot(vgm_rain1,Var.rain1)
Var.autorain1 <- fit.variogram(vgm_rain1, Var.rain1, fit.sills = TRUE, fit.ranges = TRUE)
plot(vgm_rain1, Var.autorain1)

###
blank_raster<-raster(nrow=100,ncol=100,extent(yunlin))
values(blank_raster)<- 1
bound_raster<-rasterize(yunlin,blank_raster)
bound_raster[!(is.na(bound_raster))] <- 1
grd_c<-as(bound_raster,"SpatialGridDataFrame")
###

kriged_rain1 <- krige(year2016~ 1, rain, grd_c, model=Var.rain1) # ordinary kriging
spplot(kriged_rain1["var1.pred"], key.space = "right")

###
query_point <- data.frame(x = c(186228.146,189557.6,183509.158,183653.1,187777.4,188849.138,184160.158,179617.174,180098.9,182069.84) ,
                          y = c(2623948.283,2622959,2620469.296,2617392,2620597.806,2614664.32,2611723.33,2616755.309,2613341,2613833))
coordinates(query_point) <- ~x+y
proj4string(query_point) <- CRS("+init=epsg:3826 +ellps=WGS84")
###

interpolated_rain1 <- over(query_point, kriged_rain1)
print(interpolated_rain1)


vgm_rain2 <- variogram(year2017 ~ 1, rain)
plot(vgm_rain2)
Var.rain2 <- vgm(psill=250000, "Wav", range=10000, nugget=0)
plot(vgm_rain2,Var.rain2)
Var.autorain2 <- fit.variogram(vgm_rain2, Var.rain2, fit.sills = TRUE, fit.ranges = TRUE)
plot(vgm_rain2, Var.autorain2)
kriged_rain2 <- krige(year2017~ 1, rain, grd_c, model=Var.rain2) # ordinary kriging
interpolated_rain2 <- over(query_point, kriged_rain2)
print(interpolated_rain2)

vgm_rain3 <- variogram(year2018 ~ 1, rain)
plot(vgm_rain3)
Var.rain3 <- vgm(psill=600000, "Wav", range=10000, nugget=0)
plot(vgm_rain3,Var.rain3)
Var.autorain3 <- fit.variogram(vgm_rain3, Var.rain3, fit.sills = TRUE, fit.ranges = TRUE)
plot(vgm_rain3, Var.autorain3)
kriged_rain3 <- krige(year2018~ 1, rain, grd_c, model=Var.rain3) # ordinary kriging
interpolated_rain3 <- over(query_point, kriged_rain3)
print(interpolated_rain3)

spplot(kriged_rain3["var1.pred"], key.space = "right")

vgm_rain4 <- variogram(year2019 ~ 1, rain)
plot(vgm_rain4)
Var.rain4 <- vgm(psill=250000, "Wav", range=10000, nugget=0)
plot(vgm_rain4,Var.rain4)
Var.autorain4 <- fit.variogram(vgm_rain4, Var.rain4, fit.sills = TRUE, fit.ranges = TRUE)
plot(vgm_rain4, Var.autorain4)
kriged_rain4 <- krige(year2019~ 1, rain, grd_c, model=Var.rain4) # ordinary kriging
interpolated_rain4 <- over(query_point, kriged_rain4)
print(interpolated_rain4)

vgm_rain5 <- variogram(year2020 ~ 1, rain)
plot(vgm_rain5)
Var.rain5 <- vgm(psill=250000, "Wav", range=10000, nugget=0)
plot(vgm_rain5,Var.rain5)
Var.autorain5 <- fit.variogram(vgm_rain5, Var.rain5, fit.sills = TRUE, fit.ranges = TRUE)
plot(vgm_rain5, Var.autorain5)
kriged_rain5 <- krige(year2020~ 1, rain, grd_c, model=Var.rain4) # ordinary kriging
interpolated_rain5 <- over(query_point, kriged_rain5)
print(interpolated_rain5)

#對抽水量
coordinates(pump) <- ~TWD97_X+TWD97_Y
proj4string(pump) <- CRS("+init=epsg:3826 +ellps=WGS84")
vgm_pump1 <- variogram(year2016 ~ 1, pump)
plot(vgm_pump1)
Var.pump1 <- vgm(psill=400000, "Wav", range=4000, nugget=0)
plot(vgm_pump1,Var.pump1)
Var.autopump1 <- fit.variogram(vgm_pump1, Var.pump1, fit.sills = TRUE, fit.ranges = TRUE)
plot(vgm_pump1, Var.autopump1)
kriged_pump1 <- krige(year2016~ 1, pump, grd_c, model=Var.pump1) # ordinary kriging
interpolated_pump1 <- over(query_point, kriged_pump1)
print(interpolated_pump1)

vgm_pump2 <- variogram(year2017 ~ 1, pump)
plot(vgm_pump2)
Var.pump2 <- vgm(psill=60000000, "Wav", range=5000, nugget=0)
plot(vgm_pump2,Var.pump2)
Var.autopump2 <- fit.variogram(vgm_pump2, Var.pump2, fit.sills = TRUE, fit.ranges = TRUE)
plot(vgm_pump2, Var.autopump2)
kriged_pump2 <- krige(year2017~ 1, pump, grd_c, model=Var.pump2) # ordinary kriging
interpolated_pump2 <- over(query_point, kriged_pump2)
print(interpolated_pump2)

vgm_pump3 <- variogram(year2018 ~ 1, pump)
plot(vgm_pump3)
Var.pump3 <- vgm(psill=60000000, "Wav", range=5000, nugget=0)
plot(vgm_pump3,Var.pump3)
Var.autopump3 <- fit.variogram(vgm_pump3, Var.pump3, fit.sills = TRUE, fit.ranges = TRUE)
plot(vgm_pump3, Var.autopump3)
kriged_pump3 <- krige(year2018~ 1, pump, grd_c, model=Var.pump3) # ordinary kriging
interpolated_pump3 <- over(query_point, kriged_pump3)
print(interpolated_pump3)

vgm_pump4 <- variogram(year2019 ~ 1, pump)
plot(vgm_pump4)
Var.pump4 <- vgm(psill=30000000, "Wav", range=5000, nugget=0)
plot(vgm_pump4,Var.pump4)
Var.autopump4 <- fit.variogram(vgm_pump4, Var.pump4, fit.sills = TRUE, fit.ranges = TRUE)
plot(vgm_pump4, Var.autopump4)
kriged_pump4 <- krige(year2019~ 1, pump, grd_c, model=Var.pump4) # ordinary kriging
interpolated_pump4 <- over(query_point, kriged_pump4)
print(interpolated_pump4)

vgm_pump5 <- variogram(year2020 ~ 1, pump)
plot(vgm_pump5)
Var.pump5 <- vgm(psill=30000000, "Wav", range=5000, nugget=0)
plot(vgm_pump5,Var.pump5)
Var.autopump5 <- fit.variogram(vgm_pump5, Var.pump5, fit.sills = TRUE, fit.ranges = TRUE)
plot(vgm_pump5, Var.autopump5)
kriged_pump5 <- krige(year2020~ 1, pump, grd_c, model=Var.pump5) # ordinary kriging
interpolated_pump5 <- over(query_point, kriged_pump5)
print(interpolated_pump5)


#LSTM

# 設定檔案名稱列表
files <- paste0("A", 1:10, ".xlsx")

# 記錄每個檔案的預測值
predictions <- list()

# 開始 for loop
for (file in files) {
  # 讀取資料
  data1 <- read_excel(file)
  data1 <- as.data.frame(data1)
  
  # 標準化數據
  scaled_data <- scale(data1[, c("groundwater", "rain", "pump")])
  
  # 創建輸入數據與標籤
  X <- array(scaled_data[1:4,], dim = c(1, 4, 3))  # 前四年的資料
  y <- scaled_data[5, 1]  # 第五年的地下水位作為目標值
  
  # 定義 LSTM 模型
  lstm_model <- nn_module(
    "LSTMModel",
    
    # 初始化模型
    initialize = function(input_size, hidden_size, output_size) {
      self$lstm <- nn_lstm(
        input_size = input_size,     # 輸入層大小 (三個特徵)
        hidden_size = hidden_size,   # 隱藏層大小 (隨意設置)
        batch_first = TRUE           # 指定 batch 是第一維
      )
      self$fc <- nn_linear(hidden_size, output_size)  # 全連接層，將 LSTM 輸出映射到目標輸出
    },
    
    # 前向傳播
    forward = function(x) {
      lstm_out <- self$lstm(x)  # LSTM 層，返回輸出和隱藏狀態
      out <- lstm_out[[1]]  # lstm_out 是一個列表，第一個是輸出
      out <- out[ , -1, ]  # 取最後一個時間步的輸出
      out <- self$fc(out)  # 通過全連接層
      return(out)
    }
  )
  
  # 設定模型參數
  input_size <- 3  # 三個特徵 (groundwater, rain, pump)
  hidden_size <- 50 #10,20,50,80
  output_size <- 1  # 預測值是地下水位
  
  # 初始化模型
  model <- lstm_model(input_size, hidden_size, output_size)
  
  # 定義損失函數與優化器
  criterion <- nn_mse_loss()
  optimizer <- optim_adam(model$parameters, lr = 0.001)
  
  # 轉換資料格式到 torch tensors
  X_tensor <- torch_tensor(X, dtype = torch_float32())
  y_tensor <- torch_tensor(y, dtype = torch_float32())
  
  # 訓練模型
  num_epochs <- 80 #10,20,50,80,100
  for (epoch in 1:num_epochs) {
    model$train()
    
    # 前向傳播
    output <- model(X_tensor)
    y_tensor <- y_tensor$view_as(output)
    loss <- criterion(output, y_tensor)
    
    # 反向傳播
    optimizer$zero_grad()
    loss$backward()
    optimizer$step()
    
    if (epoch %% 10 == 0) {
      cat("Epoch:", epoch, "Loss:", loss$item(), "\n")
    }
  }
  
  # 預測
  model$eval()
  predicted <- model(X_tensor)$detach()$item()
  
  # 反標準化
  predicted_groundwater <- predicted * attr(scaled_data, "scaled:scale")[1] + attr(scaled_data, "scaled:center")[1]
  
  # 打印預測結果並保存到列表中
  cat("檔案:", file, "預測的地下水位:", predicted_groundwater, "\n")
  predictions[[file]] <- predicted_groundwater
}

# 最後打印所有檔案的預測值
cat("\n所有檔案的預測結果:\n")
for (file in files) {
  station_name <- sub(".xlsx", "", file)  # 去掉檔案名中的.xlsx
  cat(station_name, "測站第五年預測的地下水位：", round(predictions[[file]], 3), "\n")
}

#畫圖
a3_hidden<-read_excel("a3_hidden.xlsx")
a3_hidden_long <- reshape2::melt(a3_hidden, id.vars = "iter", variable.name = "n", value.name = "value")
ggplot(a3_hidden_long, aes(x = iter, y = value, color = n, group = n)) +
  geom_line(linewidth = 1) +  # 使用 linewidth 而非 size
  geom_point(size = 2) +
  labs(title = "不同隱藏層", x = "Iteration", y = "Value") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# 準備檔案名稱與測站名稱
files <- paste0("a", 1:10, "_hidden.xlsx")
stations <- c("芳草(1)", "拯民", "宏崙(1)", "秀潭", "土庫(2)", "舊庄(1)", "崙子(1)", "元長(1)", "忠孝", "客厝(1)")

station_titles <- paste0(stations, " 測站不同隱藏層之均方誤差")

# for 迴圈自動生成圖表
for (i in 1:length(files)) {
  # 讀取每個檔案
  data <- read_excel(files[i])
  
  # 數據轉換為長格式
  data_long <- reshape2::melt(data, id.vars = "iter", variable.name = "n", value.name = "value")
  
  # 繪製折線圖
  p <- ggplot(data_long, aes(x = iter, y = value, color = n, group = n)) +
    geom_line(linewidth = 1) +  # 使用 linewidth 而非 size
    geom_point(size = 2) +
    labs(title = station_titles[i], x = "Iteration", y = "MSE") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # 打印圖表
  print(p)
}

# 準備檔案名稱與測站名稱
files <- paste0("a", 1:10, "_iter.xlsx")
station_titles <- paste0("A", 1:10, " 測站不同迭代次數之均方誤差")

# for 迴圈自動生成圖表
for (i in 1:length(files)) {
  # 讀取每個檔案
  data <- read_excel(files[i])
  
  # 數據轉換為長格式
  data$iter <- as.numeric(gsub("th", "", data$iter))
  data_long <- reshape2::melt(data, id.vars = "iter", variable.name = "times", value.name = "value")
  # 繪製折線圖
  p <- ggplot(data_long, aes(x = iter, y = value, color = times, group = times)) +
    geom_line(linewidth = 1) +  # 使用 linewidth 而非 size
    geom_point(size = 2) +
    labs(title = station_titles[i], x = "Iteration", y = "MSE") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # 打印圖表
  print(p)
}


######################################################
data1 <- read_excel("5station.xlsx")
coordinates(data1) <- ~ x + y

# 創建克里金模型，對 rain 進行插值
kriging_model <- gstat(formula = rain ~ 1, locations = data1, nmax = 5)

# 創建一個規則網格，設定範圍
grid <- expand.grid(x = seq(min(data1@coords[,1]), max(data1@coords[,1]), by = 0.1),
                    y = seq(min(data1@coords[,2]), max(data1@coords[,2]), by = 0.1))

coordinates(grid) <- ~ x + y

# 進行克里金插值
kriging_result <- predict(kriging_model, newdata = grid)

# 將插值結果轉換為 data frame
kriging_df <- as.data.frame(kriging_result)

# 繪製等高線圖

spplot(kriging_result, "var1.pred",
       main = "降雨分佈",
       xlab = "水平方向",ylab = list(label = "垂\n直\n座\n標", rot = 0))

data1 <- read_excel("5station.xlsx")
coordinates(data1) <- ~ x + y

# 創建克里金模型，對 rain 進行插值
kriging_model <- gstat(formula = pump ~ 1, locations = data1, nmax = 5)

# 創建一個規則網格，設定範圍
grid <- expand.grid(x = seq(min(data1@coords[,1]), max(data1@coords[,1]), by = 0.1),
                    y = seq(min(data1@coords[,2]), max(data1@coords[,2]), by = 0.1))

coordinates(grid) <- ~ x + y

# 進行克里金插值
kriging_result <- predict(kriging_model, newdata = grid)

# 將插值結果轉換為 data frame
kriging_df <- as.data.frame(kriging_result)


spplot(kriging_result, "var1.pred",
       main = "抽水量分佈",
       xlab = "水平方向",ylab = list(label = "垂\n直\n座\n標", rot = 0))

####################################
actuals_list <- c(12.08 ,6.62 ,6.35 ,-5.43  ,2.10 ,12.37,-14.32,-10.35,-8.18,-10.28)
predictions_list <- c(12.07413,6.610476 ,6.330354 ,-5.456216,2.183889,12.36948,-14.31443 ,-10.34296 ,-8.187335 ,-10.27652 )

# 計算每個測站的 MSE 和 MAE
mse_values <- (actuals_list - predictions_list)^2  # 每個測站的 MSE
mae_values <- abs(actuals_list - predictions_list)  # 每個測站的 MAE
sqrt((actuals_list - predictions_list)^2)
