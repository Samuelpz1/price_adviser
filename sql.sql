SELECT b.name , model, price , `condition`, mileage , year
FROM cars as c
JOIN brands as b
    ON b.brand_id = c.brand_id
JOIN models as m
    ON m.model_id = c.model_id
JOIN conditions AS co
    ON co.condition_id= c.condition_id
limit 5;
