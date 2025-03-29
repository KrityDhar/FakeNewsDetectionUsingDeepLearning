print("\nTraining Model...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32)


y_pred_prob = model.predict(X_test).flatten()  
y_pred = (y_pred_prob >= 0.5).astype(int)  